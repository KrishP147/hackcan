from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List
import shutil
import asyncio
import numpy as np
import uuid
from pathlib import Path
from io import BytesIO

from services import cloudinary_service, project_manager, ffmpeg_service, sam2_service
from services.auth_service import get_current_user
# film_service  # FILM disabled - using RIFE instead

load_dotenv()

app = FastAPI(title="FrameShift AI")

# Track cancellable operations per project
_cancel_flags: dict[str, bool] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-vercel-domain.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization"],
)


@app.on_event("startup")
async def startup():
    project_manager.reset_stuck_projects()


@app.get("/health")
async def health():
    return {"status": "ok"}


# --- Upload ---

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    current_user: dict | None = Depends(get_current_user),
):
    project = project_manager.create_project()
    project_dir = project_manager.get_project_dir(project["project_id"])

    # Store user_id if authenticated
    if current_user:
        project_manager.update_status(project["project_id"], user_id=current_user.get("sub"))

    video_path = project_dir / "original.mp4"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Video uploaded and staged for Cloudinary processing
    return {
        "project_id": project["project_id"],
    }


# --- Extract ---

class ExtractRequest(BaseModel):
    project_id: str

def _background_extract(project_id: str):
    """Background task: extract frames (YOLO detection disabled)."""
    project_dir = project_manager.get_project_dir(project_id)
    video_path = project_dir / "original.mp4"
    frames_dir = project_dir / "frames"

    # Extract frames at native fps, 720p working resolution
    project_manager.update_status(project_id, status="extracting")
    try:
        info = ffmpeg_service.probe_video(video_path)
        if info["duration"] > 15:
            project_manager.update_status(
                project_id, status="error",
                error="Clip too long (max 15s)")
            return
        frame_count, used_fps = ffmpeg_service.extract_frames(video_path, frames_dir)
    except Exception as e:
        print(f"[extract] FFmpeg failed: {e}")
        project_manager.update_status(project_id, status="error", error=str(e))
        return

    # Read frame dimensions from first frame
    from PIL import Image
    first_frame = sorted(frames_dir.glob("frame_*.jpg"))[0]
    img = Image.open(first_frame)
    frame_width, frame_height = img.size

    # Mark ready immediately so frontend can show frames via API
    project_manager.update_status(project_id, status="ready", frame_count=frame_count,
                                   fps=used_fps,
                                   frame_width=frame_width, frame_height=frame_height,
                                   detecting=False, detections={})

    # YOLO detection disabled
    # # Run YOLO on all frames, updating detections progressively
    # detections = {}
    # frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    # for i, f in enumerate(frame_files, start=1):
    #     dets = yolo_service.detect(f)
    #     detections[str(i)] = dets
    #     # Update every 10 frames so frontend can poll partial results
    #     if i % 10 == 0 or i == len(frame_files):
    #         project_manager.update_status(project_id, detections=detections, detected_frames=i)
    # 
    # project_manager.update_status(project_id, detecting=False, detections=detections, detected_frames=len(frame_files))

@app.post("/extract")
async def extract_frames(req: ExtractRequest, background_tasks: BackgroundTasks):
    project_manager.update_status(req.project_id, status="processing")
    background_tasks.add_task(_background_extract, req.project_id)
    return {"project_id": req.project_id, "status": "processing"}


@app.get("/project/{project_id}/status")
async def get_project_status(project_id: str):
    """Get project status, including mask count if masks exist."""
    status = project_manager.get_status(project_id)
    
    # If mask_count is not in status, check if masks exist and update count
    if "mask_count" not in status or status.get("mask_count") is None:
        project_dir = project_manager.get_project_dir(project_id)
        masks_dir = project_dir / "masks"
        if masks_dir.exists():
            existing_masks = list(masks_dir.glob("mask_*.png"))
            mask_count = len(existing_masks)
            if mask_count > 0:
                # Update status with mask count and set segment_status to "done" if not already set
                project_manager.update_status(
                    project_id,
                    mask_count=mask_count,
                    segment_status=status.get("segment_status") or "done",
                    segmenting=False,
                )
                status["mask_count"] = mask_count
                status["segment_status"] = status.get("segment_status") or "done"
                status["segmenting"] = False
    
    return status


@app.get("/frame/{project_id}/{frame_index}")
async def get_frame(project_id: str, frame_index: int):
    project_dir = project_manager.get_project_dir(project_id)
    frame_path = project_dir / "frames" / f"frame_{frame_index:04d}.jpg"
    if not frame_path.exists():
        return {"error": "Frame not found"}
    return FileResponse(frame_path, media_type="image/jpeg")


# --- Detect --- (DISABLED)

# class DetectRequest(BaseModel):
#     project_id: str
#     frame_index: int

class SegmentRequest(BaseModel):
    project_id: str
    frame_index: int
    click_x: int
    click_y: int

async def _background_segment_and_propagate(project_id: str, frame_index: int, click_x: int, click_y: int):
    """Background task: segment only the clicked frame (no propagation)."""
    try:
        project_dir = project_manager.get_project_dir(project_id)
        frames_dir = project_dir / "frames"
        frame_path = frames_dir / f"frame_{frame_index:04d}.jpg"
        masks_dir = project_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        if not frame_path.exists():
            project_manager.update_status(
                project_id, 
                segmenting=False, 
                segment_status="error",
                segment_error=f"Frame {frame_index} not found"
            )
            return

        project_manager.update_status(project_id, segmenting=True, segment_status="segmenting")

        print(f"[SAM2] Starting segmentation for frame {frame_index} at ({click_x}, {click_y})")
        
        # Segment only the clicked frame (no propagation)
        loop = asyncio.get_event_loop()
        mask = await loop.run_in_executor(
            None,
            sam2_service.segment_frame,
            frame_path,
            click_x,
            click_y
        )
        print(f"[SAM2] Segmentation complete, mask shape: {mask.shape}")

        # Save mask for this frame only
        from PIL import Image
        mask_img = (mask.astype(np.uint8)) * 255
        mask_path = masks_dir / f"mask_{frame_index:04d}.png"
        Image.fromarray(mask_img).save(mask_path)
        from services import mask_service
        mask_service.condition_single(mask_path)
        print(f"[SAM2] Saved mask to {mask_path}")

        # Count existing masks to update mask_count
        existing_masks = list(masks_dir.glob("mask_*.png"))
        mask_count = len(existing_masks)

        project_manager.update_status(
            project_id, segmenting=False, segment_status="done",
            mask_count=mask_count, anchor_frame=frame_index,
            click_x=click_x, click_y=click_y,
        )
        print(f"[SAM2] Segmentation complete for frame {frame_index}")
        print(f"[SAM2] Updated status: segmenting=False, segment_status=done, mask_count={mask_count}")
        
        # Verify status was saved correctly
        saved_status = project_manager.get_status(project_id)
        print(f"[SAM2] Status after save: segmenting={saved_status.get('segmenting')}, segment_status={saved_status.get('segment_status')}, mask_count={saved_status.get('mask_count')}")
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[SAM2] Error during segmentation: {str(e)}")
        print(error_trace)
        project_manager.update_status(
            project_id,
            segmenting=False,
            segment_status="error",
            segment_error=str(e),
        )

@app.post("/segment")
async def segment_object(req: SegmentRequest, background_tasks: BackgroundTasks):
    """Segment object at click point and propagate mask across frames."""
    print(f"[SAM2] /segment endpoint called: project={req.project_id}, frame={req.frame_index}, click=({req.click_x}, {req.click_y})")
    
    project_dir = project_manager.get_project_dir(req.project_id)
    frame_path = project_dir / "frames" / f"frame_{req.frame_index:04d}.jpg"

    if not frame_path.exists():
        print(f"[SAM2] Error: Frame not found at {frame_path}")
        return {"error": "Frame not found"}

    background_tasks.add_task(
        _background_segment_and_propagate,
        req.project_id, req.frame_index, req.click_x, req.click_y,
    )

    return {
        "project_id": req.project_id,
        "status": "processing",
        "anchor_frame": req.frame_index,
    }


@app.get("/mask/{project_id}/{mask_index}")
async def get_mask(project_id: str, mask_index: int):
    project_dir = project_manager.get_project_dir(project_id)
    mask_path = project_dir / "masks" / f"mask_{mask_index:04d}.png"
    if not mask_path.exists():
        return {"error": "Mask not found"}
    return FileResponse(mask_path, media_type="image/png")


# --- Edit ---

class EditRule(BaseModel):
    edit_type: str  # recolor, blur_region, color_pop, glow, resize, delete,
                    # move, bg_replace, replace  (§6 dispatch table)
    start_frame: int
    end_frame: int
    color: Optional[str] = None       # hex without #, e.g. "FF0000"
    scale: Optional[float] = None     # for resize
    prompt: Optional[str] = None      # for replace / bg_replace
    blur_strength: Optional[int] = None
    dx: Optional[int] = None          # for move (frame-pixel offset)
    dy: Optional[int] = None
    backend: Optional[str] = None     # "B" forces synth backend for replace

class EditRequest(BaseModel):
    project_id: str
    edit_rules: List[EditRule]



async def _background_edit(project_id: str, edit_rules: List[EditRule]):
    """Background task: route each rule through the §6 edit dispatch."""
    try:
        _cancel_flags[project_id] = False
        project_dir = project_manager.get_project_dir(project_id)
        frames_dir = project_dir / "frames"
        backups_dir = project_dir / "backups"

        frames_to_edit: set[int] = set()
        for rule in edit_rules:
            for i in range(rule.start_frame, rule.end_frame + 1):
                frames_to_edit.add(i)

        # Backup frames before editing (undo support)
        backups_dir.mkdir(exist_ok=True)
        import time
        backup_timestamp = str(int(time.time() * 1000))
        backup_dir = backups_dir / backup_timestamp
        backup_dir.mkdir(exist_ok=True)
        for idx in sorted(frames_to_edit):
            frame_path = frames_dir / f"frame_{idx:04d}.jpg"
            if frame_path.exists():
                shutil.copy2(str(frame_path), str(backup_dir / f"frame_{idx:04d}.jpg"))

        project_manager.update_status(
            project_id,
            last_backup_timestamp=backup_timestamp,
            edit_status="processing",
            edit_progress={"done": 0, "total": len(edit_rules)},
        )

        from services import edit_dispatch
        for n, rule in enumerate(edit_rules, 1):
            if _cancel_flags.get(project_id):
                project_manager.update_status(project_id, edit_status="cancelled")
                return
            await edit_dispatch.run_edit_rule(project_id, rule)
            project_manager.update_status(
                project_id, edit_progress={"done": n, "total": len(edit_rules)})

        project_manager.update_status(
            project_id, edit_status="done",
            edit_progress={"done": len(edit_rules), "total": len(edit_rules)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        project_manager.update_status(project_id, edit_status="error", edit_error=str(e))


class UndoRequest(BaseModel):
    project_id: str


@app.post("/edit/undo")
async def undo_edit(req: UndoRequest):
    """Restore frames from the last backup."""
    project_dir = project_manager.get_project_dir(req.project_id)
    frames_dir = project_dir / "frames"
    backups_dir = project_dir / "backups"
    
    status = project_manager.get_status(req.project_id)
    backup_timestamp = status.get("last_backup_timestamp")
    backup_frames = status.get("last_backup_frames", [])
    
    if not backup_timestamp or not backup_frames:
        return {"error": "No backup found to undo"}
    
    backup_dir = backups_dir / backup_timestamp
    if not backup_dir.exists():
        return {"error": "Backup directory not found"}
    
    # Restore frames from backup
    restored_count = 0
    for frame_idx in backup_frames:
        backup_path = backup_dir / f"frame_{frame_idx:04d}.jpg"
        frame_path = frames_dir / f"frame_{frame_idx:04d}.jpg"
        
        if backup_path.exists():
            shutil.copy2(str(backup_path), str(frame_path))
            restored_count += 1
    
    # Increment edit version to force refresh
    project_manager.update_status(
        req.project_id,
        edit_version=(status.get("edit_version", 0) or 0) + 1,
        last_backup_timestamp=None,  # Clear backup after undo
        last_backup_frames=[],
    )
    
    return {
        "status": "success",
        "restored_frames": restored_count,
        "message": f"Restored {restored_count} frame(s)"
    }


@app.post("/edit")
async def edit_frames(req: EditRequest):
    project_dir = project_manager.get_project_dir(req.project_id)
    if not any((project_dir / "frames").glob("frame_*.jpg")):
        return {"error": "No frames found. Run /extract first."}

        project_manager.update_status(
            req.project_id,
            edit_status="editing",
            edit_progress={"done": 0, "total": 0},
        )
    # Run as a proper async task instead of BackgroundTasks (which can't await)
    asyncio.ensure_future(_background_edit(req.project_id, req.edit_rules))
    return {"project_id": req.project_id, "edit_status": "uploading"}


class CancelRequest(BaseModel):
    project_id: str


@app.post("/edit/cancel")
async def cancel_edit(req: CancelRequest):
    """Cancel any running edit/refine/propagate operation."""
    _cancel_flags[req.project_id] = True
    project_manager.update_status(
        req.project_id,
        edit_status="cancelled",
        refine_status="cancelled",
        ai_edit_status="cancelled",
    )
    return {"status": "cancelled"}


class RenderRequest(BaseModel):
    project_id: str

@app.post("/render")
async def render_video(req: RenderRequest):
    project_dir = project_manager.get_project_dir(req.project_id)
    frames_dir = project_dir / "frames"
    edited_dir = project_dir / "edited"
    output_path = project_dir / "output.mp4"

    status = project_manager.get_status(req.project_id)
    fps = status.get("fps") or 30

    # Check if AI edits are done - use frames_dir, otherwise use edited_dir
    if status.get("ai_edit_status") == "done":
        # Use frames directory (contains AI-edited frames)
        ffmpeg_service.encode_video(frames_dir, output_path, fps=fps)
    else:
        # Check if regular edits are done
        if status.get("edit_status") not in ("done", None, "idle"):
            return {"error": f"Edit not complete. Current edit_status: {status.get('edit_status')}"}

        edited_frames = sorted(edited_dir.glob("frame_*.jpg"))
        if len(edited_frames) == 0:
            # No edits - use original frames
            ffmpeg_service.encode_video(frames_dir, output_path, fps=fps)
        else:
            ffmpeg_service.encode_video(edited_dir, output_path, fps=fps)

    # Return local file path
    return {
        "project_id": req.project_id,
        "video_path": str(output_path),
    }

@app.get("/render/{project_id}/video")
async def get_rendered_video(project_id: str):
    """Serve the rendered video file."""
    project_dir = project_manager.get_project_dir(project_id)
    output_path = project_dir / "output.mp4"
    
    if not output_path.exists():
        return {"error": "Video not found. Run /render first."}
    
    return FileResponse(output_path, media_type="video/mp4")
