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

from services import cloudinary_service, project_manager, ffmpeg_service, yolo_service, sam2_service, gemini_service, rife_service, storage_service
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

    # Extract frames
    project_manager.update_status(project_id, status="extracting")
    try:
        frame_count = ffmpeg_service.extract_frames(video_path, frames_dir)
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
                                   frame_width=frame_width, frame_height=frame_height,
                                   detecting=False, detections={})

    # Upload frames to Supabase Storage for persistent access
    try:
        storage_base_url = storage_service.upload_frames(project_id, frames_dir)
        if storage_base_url:
            project_manager.update_status(project_id, storage_base_url=storage_base_url)
            print(f"[extract] Frames uploaded to Supabase Storage: {storage_base_url}")
    except Exception as e:
        print(f"[extract] Supabase Storage upload failed (non-fatal): {e}")

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

# @app.post("/detect")
# async def detect_objects(req: DetectRequest):
#     project_dir = project_manager.get_project_dir(req.project_id)
#     frame_path = project_dir / "frames" / f"frame_{req.frame_index:04d}.jpg"
# 
#     if not frame_path.exists():
#         return {"error": "Frame not found"}
# 
#     detections = yolo_service.detect(frame_path)
#     return {"project_id": req.project_id, "frame_index": req.frame_index, "objects": detections}


# --- Segment ---

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
        print(f"[SAM2] Saved mask to {mask_path}")

        # Count existing masks to update mask_count
        existing_masks = list(masks_dir.glob("mask_*.png"))
        mask_count = len(existing_masks)

        project_manager.update_status(
            project_id, segmenting=False, segment_status="done",
            mask_count=mask_count, anchor_frame=frame_index,
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
    edit_type: str  # core: recolor, resize, replace, add, delete
                    # extras: bg_remove, bg_replace, gen_fill, enhance, upscale,
                    #         restore, blur, blur_region, drop_shadow, gen_recolor
    start_frame: int
    end_frame: int
    color: Optional[str] = None       # hex without #, e.g. "FF0000"
    scale: Optional[float] = None     # for resize
    prompt: Optional[str] = None      # for replace, add, bg_replace, gen_fill, gen_recolor
    blur_strength: Optional[int] = None  # for blur (default 500)
    asset_x: Optional[int] = None     # for add positioning
    asset_y: Optional[int] = None
    asset_w: Optional[int] = None
    asset_h: Optional[int] = None

class EditRequest(BaseModel):
    project_id: str
    edit_rules: List[EditRule]

def _save_edited_frame(frame_path: Path, edited_bytes: bytes):
    """Save edited bytes to frame, resizing to match original dimensions if needed."""
    from PIL import Image
    orig = Image.open(frame_path)
    orig_size = orig.size
    orig.close()
    edited = Image.open(BytesIO(edited_bytes))
    if edited.size != orig_size:
        print(f"[RESIZE] {frame_path.name}: {edited.size} -> {orig_size}")
        edited = edited.resize(orig_size, Image.LANCZOS)
    buf = BytesIO()
    edited.save(buf, format="JPEG", quality=95)
    frame_path.write_bytes(buf.getvalue())


def _composite_with_mask(original_path, edited_path, mask_array):
    """Blend edited image onto original using SAM2 mask: edited where mask=white, original elsewhere."""
    from PIL import Image
    original = np.array(Image.open(original_path).convert("RGB"))
    edited = np.array(Image.open(edited_path).convert("RGB"))

    # Resize mask to match frame dimensions if needed
    mask = mask_array
    if mask.shape[:2] != original.shape[:2]:
        mask_img = Image.fromarray(mask).resize((original.shape[1], original.shape[0]), Image.NEAREST)
        mask = np.array(mask_img)

    # Normalize mask to 0-1 float, expand to 3 channels
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    alpha = (mask > 0).astype(np.float32)[:, :, np.newaxis]

    # Resize edited to match original if Cloudinary changed dimensions
    if edited.shape[:2] != original.shape[:2]:
        edited_img = Image.fromarray(edited).resize((original.shape[1], original.shape[0]), Image.LANCZOS)
        edited = np.array(edited_img)

    result = (alpha * edited + (1 - alpha) * original).astype(np.uint8)
    Image.fromarray(result).save(str(edited_path), quality=95)


def _apply_recolor_local(frame_path, mask_array, color_hex: str):
    """Apply color tint locally using SAM2 mask before Cloudinary upload."""
    from PIL import Image
    original = np.array(Image.open(frame_path).convert("RGB"))

    mask = mask_array
    if mask.shape[:2] != original.shape[:2]:
        mask_img = Image.fromarray(mask).resize((original.shape[1], original.shape[0]), Image.NEAREST)
        mask = np.array(mask_img)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    alpha = (mask > 0).astype(np.float32)[:, :, np.newaxis]

    # Parse hex color
    color_hex = color_hex.lstrip("#")
    r, g, b = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)
    tint = np.full_like(original, [r, g, b])

    # Blend: 60% original + 40% tint in masked region
    tinted = (0.6 * original + 0.4 * tint).clip(0, 255).astype(np.uint8)
    result = (alpha * tinted + (1 - alpha) * original).astype(np.uint8)

    save_path = frame_path
    Image.fromarray(result).save(str(save_path), quality=95)
    return save_path


# Edits are processed locally then delivered via Cloudinary
from services import local_edit_service

# Edits that require a mask (object-specific)
MASK_EDITS = {"delete", "replace", "resize", "blur_region", "gen_recolor", "recolor"}
# Edits that affect the whole frame
FRAME_EDITS = {"bg_remove", "bg_replace", "gen_fill", "enhance", "upscale", "restore", "blur", "drop_shadow"}


async def _background_edit(project_id: str, edit_rules: List[EditRule]):
    """Background task: apply edits and upload results to Cloudinary."""
    try:
        from PIL import Image
        _cancel_flags[project_id] = False

        project_dir = project_manager.get_project_dir(project_id)
        frames_dir = project_dir / "frames"
        masks_dir = project_dir / "masks"
        backups_dir = project_dir / "backups"

        # Collect unique frame indices to edit
        frames_to_edit: set[int] = set()
        for rule in edit_rules:
            for i in range(rule.start_frame, rule.end_frame + 1):
                frames_to_edit.add(i)

        # Backup frames before editing
        backups_dir.mkdir(exist_ok=True)
        import time
        backup_timestamp = str(int(time.time() * 1000))
        backup_dir = backups_dir / backup_timestamp
        backup_dir.mkdir(exist_ok=True)
        
        for idx in sorted(frames_to_edit):
            frame_path = frames_dir / f"frame_{idx:04d}.jpg"
            if frame_path.exists():
                backup_path = backup_dir / f"frame_{idx:04d}.jpg"
                shutil.copy2(str(frame_path), str(backup_path))
        
        # Store backup info in status
        project_manager.update_status(
            project_id,
            last_backup_timestamp=backup_timestamp,
            last_backup_frames=list(frames_to_edit)
        )

        total = len(frames_to_edit)
        project_manager.update_status(project_id, edit_status="editing", edit_progress={"done": 0, "total": total})

        completed = 0

        for idx in sorted(frames_to_edit):
            if _cancel_flags.get(project_id):
                project_manager.update_status(project_id, edit_status="cancelled")
                return

            frame_path = frames_dir / f"frame_{idx:04d}.jpg"
            if not frame_path.exists():
                completed += 1
                project_manager.update_status(project_id, edit_progress={"done": completed, "total": total})
                continue

            mask_path = masks_dir / f"mask_{idx:04d}.png"
            has_mask = mask_path.exists()

            for rule in edit_rules:
                if rule.start_frame <= idx <= rule.end_frame:
                    t = rule.edit_type
                    print(f"[Edit] Frame {idx}: type={t}, has_mask={has_mask}")

                    # ── Object edits (require mask) ──
                    if t in MASK_EDITS:
                        if not has_mask:
                            print(f"[Edit] Skipping {t} - no mask for frame {idx}")
                            break
                            
                        if t == "recolor":
                            local_edit_service.apply_recolor(frame_path, mask_path, rule.color or "FF0000")
                        elif t == "blur_region":
                            local_edit_service.apply_blur_region(frame_path, mask_path, strength=10)
                        elif t == "resize":
                            local_edit_service.apply_resize(frame_path, mask_path, rule.scale or 1.5)
                        elif t == "delete":
                            local_edit_service.apply_remove(frame_path, mask_path)
                        elif t == "replace":
                            # Use Gemini AI to replace object
                            try:
                                edited_bytes = await gemini_service.edit_frame(
                                    frame_path,
                                    f"Replace the selected object with {rule.prompt or 'something similar'}",
                                    mask_path=mask_path
                                )
                                # Save edited frame
                                _save_edited_frame(frame_path, edited_bytes)
                            except Exception as e:
                                print(f"[Edit] Gemini replace failed for frame {idx}: {e}")
                        elif t == "gen_recolor":
                            # Use Gemini AI for AI-powered recolor
                            try:
                                color_desc = f"Change the color to {rule.color or 'FF0000'}"
                                prompt = f"{rule.prompt or 'the selected object'}, {color_desc}"
                                edited_bytes = await gemini_service.edit_frame(
                                    frame_path,
                                    prompt,
                                    mask_path=mask_path
                                )
                                _save_edited_frame(frame_path, edited_bytes)
                            except Exception as e:
                                print(f"[Edit] Gemini AI recolor failed for frame {idx}: {e}")
                                # Fallback to simple recolor
                                if rule.color:
                                    local_edit_service.apply_recolor(frame_path, mask_path, rule.color)

                    # ── Whole-frame edits ──
                    elif t in FRAME_EDITS:
                        if t == "upscale":
                            # Upscale only processes key frames (already filtered in frames_to_edit)
                            local_edit_service.apply_upscale(frame_path, scale=2)
                            print(f"[Edit] Upscaled key frame {idx}")
                        elif t == "enhance":
                            local_edit_service.apply_enhance(frame_path)
                        elif t == "restore":
                            local_edit_service.apply_restore(frame_path)
                        elif t == "blur":
                            local_edit_service.apply_blur(frame_path, strength=rule.blur_strength or 10)
                        elif t == "bg_remove":
                            # Use Gemini AI to remove background
                            try:
                                edited_bytes = await gemini_service.edit_frame(
                                    frame_path,
                                    "Remove the background, keep only the main subject",
                                    mask_path=None
                                )
                                _save_edited_frame(frame_path, edited_bytes)
                            except Exception as e:
                                print(f"[Edit] Gemini bg_remove failed for frame {idx}: {e}")
                        elif t == "bg_replace":
                            # Use Gemini AI to replace background
                            try:
                                edited_bytes = await gemini_service.edit_frame(
                                    frame_path,
                                    f"Replace the background with {rule.prompt or 'a studio background'}",
                                    mask_path=None
                                )
                                _save_edited_frame(frame_path, edited_bytes)
                            except Exception as e:
                                print(f"[Edit] Gemini bg_replace failed for frame {idx}: {e}")
                        elif t == "gen_fill":
                            # Use Gemini AI for generative fill
                            try:
                                edited_bytes = await gemini_service.edit_frame(
                                    frame_path,
                                    rule.prompt or "Fill the empty space naturally",
                                    mask_path=None
                                )
                                _save_edited_frame(frame_path, edited_bytes)
                            except Exception as e:
                                print(f"[Edit] Gemini gen_fill failed for frame {idx}: {e}")

                    break

            completed += 1
            project_manager.update_status(project_id, edit_progress={"done": completed, "total": total})

        project_manager.update_status(project_id, edit_status="done", edit_progress={"done": total, "total": total})

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


class RefineRequest(BaseModel):
    project_id: str
    frame_index: int  # 1-based
    prompt: str = ""  # Optional extra context
    start_frame: int = 0  # 0 = use frame_index only, otherwise process range
    end_frame: int = 0  # 0 = use frame_index only


@app.post("/edit/refine")
async def refine_frame(req: RefineRequest):
    """Use Gemini AI (nanobanana) to make frames look completely photorealistic.
    If start_frame and end_frame are provided, processes only start and end frames within that range.
    Otherwise, only processes the frame_index."""
    project_dir = project_manager.get_project_dir(req.project_id)
    frames_dir = project_dir / "frames"
    masks_dir = project_dir / "masks"

    # Determine which frames to process (only start and end within boundaries)
    if req.start_frame > 0 and req.end_frame > 0 and req.end_frame >= req.start_frame:
        # Process only start and end frames within the range
        frames_to_process = [req.start_frame]
        if req.end_frame != req.start_frame:
            frames_to_process.append(req.end_frame)
    else:
        # Process only the current frame
        frames_to_process = [req.frame_index]

    # Validate frames exist
    for frame_idx in frames_to_process:
        frame_path = frames_dir / f"frame_{frame_idx:04d}.jpg"
        if not frame_path.exists():
            return {"error": f"Frame {frame_idx} not found"}

    project_manager.update_status(req.project_id, refine_status="processing", refine_progress={"done": 0, "total": len(frames_to_process)})

    async def _background_refine():
        try:
            backups_dir = project_dir / "backups"
            backups_dir.mkdir(exist_ok=True)
            import time
            backup_timestamp = str(int(time.time() * 1000))
            backup_dir = backups_dir / backup_timestamp
            backup_dir.mkdir(exist_ok=True)
            
            backup_frames = []
            completed = 0
            
            for frame_idx in frames_to_process:
                frame_path = frames_dir / f"frame_{frame_idx:04d}.jpg"
                mask_path = masks_dir / f"mask_{frame_idx:04d}.png"
                
                # Backup frame before refining
                backup_path = backup_dir / f"frame_{frame_idx:04d}.jpg"
                shutil.copy2(str(frame_path), str(backup_path))
                backup_frames.append(frame_idx)
                
                # Check if mask exists - if so, enhance only the segmented object
                has_mask = mask_path.exists() if mask_path else False
                
                if has_mask:
                    # Prompt to enhance only the segmented object while keeping the rest of the frame unchanged
                    prompt = (
                        f"Enhance only the selected/segmented object in this image to look completely photorealistic. "
                        f"Apply realistic textures, natural lighting, proper shadows, reflections, and depth ONLY to the "
                        f"segmented object. Keep the rest of the frame exactly as it is - do not change anything outside "
                        f"the selected object. The background and other objects should remain completely unchanged. "
                        f"Make the segmented object look like it was captured by a professional camera with realistic details, "
                        f"but preserve the original structure and composition of the entire image. "
                        f"{req.prompt if req.prompt else ''}"
                    ).strip()
                    # Apply enhancement only to the masked region
                    edited_bytes = await gemini_service.edit_frame(frame_path, prompt, mask_path=mask_path)
                else:
                    # No mask - enhance entire frame (fallback behavior)
                    prompt = (
                        f"Transform this entire image into a completely photorealistic photograph. "
                        f"Enhance every single object, person, and element in the frame to look like a high-quality, "
                        f"professional photograph with natural lighting, realistic textures, proper shadows, reflections, "
                        f"and depth. Make all objects in the scene look realistic and natural - enhance each one individually "
                        f"while maintaining the overall composition. Make it look like it was captured by a professional camera. "
                        f"Keep the same composition and subject matter, but make everything look more realistic, detailed, and natural. "
                        f"{req.prompt if req.prompt else ''}"
                    ).strip()
                    # Apply enhancement to the entire frame
                    edited_bytes = await gemini_service.edit_frame(frame_path, prompt, mask_path=None)
                _save_edited_frame(frame_path, edited_bytes)
                
                completed += 1
                project_manager.update_status(
                    req.project_id,
                    refine_progress={"done": completed, "total": len(frames_to_process)},
                )
            
            project_manager.update_status(
                req.project_id,
                last_backup_timestamp=backup_timestamp,
                last_backup_frames=backup_frames,
                refine_status="done",
                edit_version=(project_manager.get_status(req.project_id).get("edit_version", 0) or 0) + 1,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            project_manager.update_status(
                req.project_id,
                refine_status="error",
                refine_error=str(e),
            )

    asyncio.ensure_future(_background_refine())
    return {"status": "processing"}


class ChangeLogEntry(BaseModel):
    id: str
    projectId: str
    timestamp: int
    type: str  # "segment" | "edit" | "refine"
    frameIndex: int
    data: dict

class PropagateRequest(BaseModel):
    project_id: str
    frame_index: int  # The edited reference frame (1-based)
    prompt: str       # Description of the edit to propagate
    start_frame: int = 1
    end_frame: int = 0  # 0 = last frame
    interval: int = 8
    change_logs: list[ChangeLogEntry] = []  # All logged changes to replay


@app.post("/edit/propagate")
async def propagate_edit(req: PropagateRequest):
    """Propagate all logged changes to every 15th frame using RIFE interpolation."""
    project_dir = project_manager.get_project_dir(req.project_id)
    frames_dir = project_dir / "frames"

    # Use the specified range - if end_frame is 0 or same as start, use only start frame
    start_frame = req.start_frame
    end_frame = req.end_frame
    
    # If end_frame is 0 or not set, only process the start frame
    if end_frame == 0 or end_frame < start_frame:
        end_frame = start_frame
    
    # Ensure we don't exceed available frames
    status = project_manager.get_status(req.project_id)
    max_frames = status.get("frame_count", len(list(frames_dir.glob("frame_*.jpg"))))
    end_frame = min(end_frame, max_frames)
    start_frame = min(start_frame, max_frames)

    project_manager.update_status(
        req.project_id,
        ai_edit_status="processing",
        ai_edit_phase="transforming",
        ai_edit_progress={"done": 0, "total": 0},
        ai_interpolation_progress={"done": 0, "total": 0},
        ai_generation_id=None,
    )

    # Process change logs sequentially - only within the specified range
    asyncio.ensure_future(_background_propagate_changes(
        req.project_id,
        req.change_logs or [],
        start_frame,
        end_frame,
        interval=15,  # Apply to every 15th frame
    ))

    return {"project_id": req.project_id, "status": "processing"}


# --- AI Edit ---

class AIPreviewRequest(BaseModel):
    project_id: str
    frame_index: int
    prompt: str

class AIAcceptRequest(BaseModel):
    project_id: str
    generation_id: str
    start_frame: int
    end_frame: int
    interval: int = 15  # Gen AI every Nth frame between start and end

class AIRejectRequest(BaseModel):
    project_id: str
    generation_id: str

class AIRetryRequest(BaseModel):
    project_id: str
    generation_id: str

@app.post("/ai/edit/preview")
async def ai_edit_preview(req: AIPreviewRequest):
    """Generate a preview of AI edit on a single frame."""
    project_dir = project_manager.get_project_dir(req.project_id)
    frames_dir = project_dir / "frames"
    masks_dir = project_dir / "masks"
    previews_dir = project_dir / "previews"
    previews_dir.mkdir(exist_ok=True)

    frame_path = frames_dir / f"frame_{req.frame_index:04d}.jpg"
    if not frame_path.exists():
        return {"error": f"Frame {req.frame_index} not found"}

    # Load mask for preview frame if available
    mask_path = masks_dir / f"mask_{req.frame_index:04d}.png"
    mask_path_param = mask_path if mask_path.exists() else None

    # Generate preview using Gemini
    try:
        print(f"Generating preview for frame {req.frame_index} with prompt: {req.prompt}")
        if mask_path_param:
            print(f"Using mask for preview: {mask_path_param}")
        preview_bytes = await gemini_service.edit_frame(frame_path, req.prompt, mask_path=mask_path_param)
        print(f"Preview generated successfully, size: {len(preview_bytes)} bytes")
        
        # Save preview
        generation_id = str(uuid.uuid4())
        preview_path = previews_dir / f"preview_{generation_id}.jpg"
        preview_path.write_bytes(preview_bytes)
        print(f"Preview saved to: {preview_path}")

        # Store generation metadata
        project_manager.update_status(
            req.project_id,
            ai_generation_id=generation_id,
            ai_preview_url=f"/preview/{req.project_id}/{generation_id}",
            ai_prompt=req.prompt,
            ai_original_frame=req.frame_index,
            ai_edit_status="preview",
        )

        return {
            "generation_id": generation_id,
            "preview_url": f"/preview/{req.project_id}/{generation_id}",
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error generating preview: {str(e)}")
        print(error_trace)
        return {"error": f"{str(e)}"}


@app.get("/preview/{project_id}/{generation_id}")
async def get_preview(project_id: str, generation_id: str):
    """Serve preview image."""
    from fastapi import HTTPException
    project_dir = project_manager.get_project_dir(project_id)
    preview_path = project_dir / "previews" / f"preview_{generation_id}.jpg"
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(str(preview_path), media_type="image/jpeg")


@app.post("/ai/edit/accept")
async def ai_edit_accept(req: AIAcceptRequest):
    """Accept preview and propagate transformation to all frames."""
    print(f"Accept endpoint called: project_id={req.project_id}, generation_id={req.generation_id}")
    project_dir = project_manager.get_project_dir(req.project_id)
    status = project_manager.get_status(req.project_id)

    current_status = status.get("ai_edit_status")
    if current_status == "processing":
        return {"error": "Edit already in progress", "status": "processing"}

    stored_generation_id = status.get("ai_generation_id")
    if stored_generation_id and stored_generation_id != req.generation_id:
        return {"error": "Invalid generation_id"}
    if stored_generation_id is None:
        return {"error": "This preview was already processed"}

    previews_dir = project_dir / "previews"
    preview_path = previews_dir / f"preview_{req.generation_id}.jpg"
    if not preview_path.exists():
        return {"error": "Preview not found"}

    prompt = status.get("ai_prompt", "") or "Apply the same visual edit consistently"

    project_manager.update_status(
        req.project_id,
        ai_edit_status="processing",
        ai_edit_progress={"done": 0, "total": 0},
        ai_generation_id=None,
    )

    asyncio.ensure_future(_background_ai_edit(
        req.project_id,
        req.generation_id,
        preview_path,
        prompt,
        req.start_frame,
        req.end_frame,
        req.interval,
    ))

    return {"status": "processing"}


@app.post("/ai/edit/reject")
async def ai_edit_reject(req: AIRejectRequest):
    """Reject preview and clear it."""
    project_manager.update_status(
        req.project_id,
        ai_generation_id=None,
        ai_preview_url=None,
        ai_prompt=None,
        ai_edit_status="idle",
    )
    return {"status": "rejected"}


@app.post("/ai/edit/retry")
async def ai_edit_retry(req: AIRetryRequest):
    """Retry generation with same prompt."""
    project_dir = project_manager.get_project_dir(req.project_id)
    status = project_manager.get_status(req.project_id)
    
    if status.get("ai_generation_id") != req.generation_id:
        return {"error": "Invalid generation_id"}
    
    prompt = status.get("ai_prompt", "")
    if not prompt:
        return {"error": "Prompt not found"}
    
    # Get the original frame index from status
    frame_index = status.get("ai_original_frame", 1)
    
    frames_dir = project_dir / "frames"
    frame_path = frames_dir / f"frame_{frame_index:04d}.jpg"
    if not frame_path.exists():
        return {"error": f"Frame {frame_index} not found"}

    try:
        preview_bytes = await gemini_service.edit_frame(frame_path, prompt)
        
        # Generate new generation ID
        new_generation_id = str(uuid.uuid4())
        previews_dir = project_dir / "previews"
        previews_dir.mkdir(exist_ok=True)
        preview_path = previews_dir / f"preview_{new_generation_id}.jpg"
        preview_path.write_bytes(preview_bytes)

        project_manager.update_status(
            req.project_id,
            ai_generation_id=new_generation_id,
            ai_preview_url=f"/preview/{req.project_id}/{new_generation_id}",
            ai_original_frame=frame_index,
            ai_edit_status="preview",
        )

        return {
            "generation_id": new_generation_id,
            "preview_url": f"/preview/{req.project_id}/{new_generation_id}",
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


async def _background_propagate_changes(
    project_id: str,
    change_logs: list[ChangeLogEntry],
    start_frame: int,
    end_frame: int,
    interval: int = 15,
):
    """Background task: Apply all logged changes sequentially to every Nth frame, then interpolate with RIFE."""
    try:
        project_dir = project_manager.get_project_dir(project_id)
        frames_dir = project_dir / "frames"
        masks_dir = project_dir / "masks"
        
        # Calculate key frames: only start and end frames
        key_frames = []
        
        # Always include start frame if it exists
        start_path = frames_dir / f"frame_{start_frame:04d}.jpg"
        if start_path.exists():
            key_frames.append(start_frame)
        
        # Always include end frame if it's different from start and exists
        if end_frame != start_frame:
            end_path = frames_dir / f"frame_{end_frame:04d}.jpg"
            if end_path.exists():
                key_frames.append(end_frame)
        
        # Sort to ensure correct order
        key_frames = sorted(set(key_frames))
        
        total_logs = len(change_logs)
        total_key_frames = len(key_frames)
        total_operations = total_logs * total_key_frames
        
        project_manager.update_status(
            project_id,
            ai_edit_phase="transforming",
            ai_edit_progress={"done": 0, "total": total_operations},
            ai_interpolation_progress={"done": 0, "total": 0},
        )
        
        completed = 0
        
        # Process each change log sequentially
        for log_idx, log_entry in enumerate(change_logs):
            print(f"[Propagate] Processing change log {log_idx + 1}/{total_logs}: {log_entry.type}")
            
            # Apply this change to each key frame
            for frame_idx in key_frames:
                frame_path = frames_dir / f"frame_{frame_idx:04d}.jpg"
                if not frame_path.exists():
                    completed += 1
                    project_manager.update_status(
                        project_id,
                        ai_edit_progress={"done": completed, "total": total_operations},
                    )
                    continue
                
                try:
                    if log_entry.type == "segment":
                        # Re-segment at the logged click point
                        click_x = log_entry.data.get("clickX", 0)
                        click_y = log_entry.data.get("clickY", 0)
                        mask_path = masks_dir / f"mask_{frame_idx:04d}.png"
                        
                        # Segment this frame (simplified - in production you'd use SAM2 propagation)
                        # For now, copy mask from original frame if available
                        original_mask = masks_dir / f"mask_{log_entry.frameIndex + 1:04d}.png"
                        if original_mask.exists() and original_mask != mask_path:
                            shutil.copy2(str(original_mask), str(mask_path))
                    
                    elif log_entry.type == "edit":
                        # Apply the edit
                        edit_type = log_entry.data.get("editType")
                        mask_path = masks_dir / f"mask_{frame_idx:04d}.png"
                        has_mask = mask_path.exists()
                        
                        if edit_type in MASK_EDITS and not has_mask:
                            print(f"[Propagate] Skipping {edit_type} - no mask for frame {frame_idx}")
                            completed += 1
                            project_manager.update_status(
                                project_id,
                                ai_edit_progress={"done": completed, "total": total_operations},
                            )
                            continue
                        
                        if edit_type == "recolor":
                            local_edit_service.apply_recolor(frame_path, mask_path, log_entry.data.get("color", "FF0000"))
                        elif edit_type == "blur_region":
                            local_edit_service.apply_blur_region(frame_path, mask_path, strength=10)
                        elif edit_type == "resize":
                            local_edit_service.apply_resize(frame_path, mask_path, log_entry.data.get("scale", 1.5))
                        elif edit_type == "delete":
                            local_edit_service.apply_remove(frame_path, mask_path)
                        elif edit_type == "replace":
                            edited_bytes = await gemini_service.edit_frame(
                                frame_path,
                                log_entry.data.get("prompt", "Replace object"),
                                mask_path=mask_path if has_mask else None,
                            )
                            _save_edited_frame(frame_path, edited_bytes)
                    
                    elif log_entry.type == "refine":
                        # Apply refine (make realistic)
                        mask_path = masks_dir / f"mask_{frame_idx:04d}.png"
                        has_mask = mask_path.exists()
                        
                        prompt = log_entry.data.get("prompt", "Make this look completely photorealistic")
                        if has_mask:
                            prompt = f"{prompt}. Only enhance the segmented object, keep the rest unchanged."
                        
                        edited_bytes = await gemini_service.edit_frame(
                            frame_path,
                            prompt,
                            mask_path=mask_path if has_mask else None,
                        )
                        _save_edited_frame(frame_path, edited_bytes)
                    
                    completed += 1
                    project_manager.update_status(
                        project_id,
                        ai_edit_progress={"done": completed, "total": total_operations},
                    )
                    
                except Exception as e:
                    print(f"[Propagate] Error applying {log_entry.type} to frame {frame_idx}: {e}")
                    completed += 1
                    project_manager.update_status(
                        project_id,
                        ai_edit_progress={"done": completed, "total": total_operations},
                    )
        
        print(f"[Propagate] All changes applied to key frames. Starting RIFE interpolation.")
        
        # Interpolate between key frames using RIFE (only within [start_frame, end_frame])
        total_interpolation_frames = 0
        interpolation_segments = []
        for i in range(len(key_frames) - 1):
            start_frame_idx = key_frames[i]
            end_frame_idx = key_frames[i + 1]
            frames_to_interpolate = []
            for frame_idx in range(start_frame_idx + 1, end_frame_idx):
                if frame_idx < start_frame or frame_idx > end_frame:
                    continue
                frame_path = frames_dir / f"frame_{frame_idx:04d}.jpg"
                if frame_path.exists():
                    frames_to_interpolate.append(frame_path)
            if len(frames_to_interpolate) > 0:
                frames_to_interpolate.sort(key=lambda p: int(p.stem.replace("frame_", "")) if p.stem.startswith("frame_") else 0)
                total_interpolation_frames += len(frames_to_interpolate)
                interpolation_segments.append((start_frame_idx, end_frame_idx, frames_to_interpolate))
        
        if total_interpolation_frames > 0:
            project_manager.update_status(
                project_id,
                ai_edit_phase="interpolating",
                ai_interpolation_progress={"done": 0, "total": total_interpolation_frames},
            )
            
            async def interpolate_segment(start_frame_idx: int, end_frame_idx: int, frames_to_interpolate: list):
                if len(frames_to_interpolate) == 0:
                    return
                start_path = frames_dir / f"frame_{start_frame_idx:04d}.jpg"
                end_path = frames_dir / f"frame_{end_frame_idx:04d}.jpg"
                if not start_path.exists() or not end_path.exists():
                    return
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda sp=start_path, ep=end_path, fl=frames_to_interpolate: rife_service.interpolate_pair(sp, ep, fl),
                )
                status = project_manager.get_status(project_id)
                current_done = status.get("ai_interpolation_progress", {}).get("done", 0)
                project_manager.update_status(
                    project_id,
                    ai_interpolation_progress={"done": current_done + len(frames_to_interpolate), "total": total_interpolation_frames},
                )
            
            for start_frame_idx, end_frame_idx, frames_to_interpolate in interpolation_segments:
                await interpolate_segment(start_frame_idx, end_frame_idx, frames_to_interpolate)
            print(f"[Propagate] RIFE interpolation complete: {total_interpolation_frames} frames")
        
        project_manager.update_status(
            project_id,
            ai_edit_status="done",
            ai_edit_phase="done",
            ai_edit_progress={"done": total_operations, "total": total_operations},
            ai_interpolation_progress={"done": total_interpolation_frames, "total": total_interpolation_frames},
        )
        print(f"[Propagate] Complete: applied {total_logs} changes to {total_key_frames} key frames, interpolated {total_interpolation_frames} in-between frames")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        project_manager.update_status(
            project_id,
            ai_edit_status="error",
            ai_edit_error=str(e),
        )


async def _background_ai_edit(
    project_id: str,
    generation_id: str,
    preview_path: Path,
    prompt: str,
    start_frame: int,
    end_frame: int,
    interval: int,
):
    """Background task: Apply AI transformation to start frame, every Nth frame, and end frame."""
    try:
        project_dir = project_manager.get_project_dir(project_id)
        frames_dir = project_dir / "frames"
        masks_dir = project_dir / "masks"
        
        # Key frames only within [start_frame, end_frame]: always start, then every interval-th, then end (only existing)
        key_frames = []
        # Always include start frame if it exists (so the selected start is never skipped)
        start_path = frames_dir / f"frame_{start_frame:04d}.jpg"
        if start_path.exists() and start_frame <= end_frame:
            key_frames.append(start_frame)
        for frame_idx in range(start_frame, end_frame + 1, interval):
            if frame_idx in key_frames:
                continue
            frame_path = frames_dir / f"frame_{frame_idx:04d}.jpg"
            if frame_path.exists():
                key_frames.append(frame_idx)
        # Ensure end frame is included if different from start and exists
        if end_frame not in key_frames and start_frame <= end_frame:
            end_path = frames_dir / f"frame_{end_frame:04d}.jpg"
            if end_path.exists():
                key_frames.append(end_frame)
        # Only transform within start and end (defensive filter), then dedupe and sort
        key_frames = sorted(set(f for f in key_frames if start_frame <= f <= end_frame))
        
        total = len(key_frames)
        project_manager.update_status(
            project_id,
            ai_edit_phase="transforming",
            ai_edit_progress={"done": 0, "total": total},
            ai_interpolation_progress={"done": 0, "total": 0},
        )
        
        # Process frames concurrently with a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(4)  # Process up to 4 frames concurrently
        completed_count = 0
        import time
        
        async def transform_frame(frame_idx: int, idx: int):
            """Transform a single frame with concurrency control."""
            nonlocal completed_count
            frame_path = frames_dir / f"frame_{frame_idx:04d}.jpg"
            if not frame_path.exists():
                completed_count += 1
                project_manager.update_status(
                    project_id,
                    ai_edit_progress={"done": completed_count, "total": total},
                )
                return
            
            # Print before acquiring semaphore to show when task starts
            start_time = time.time()
            print(f"[START] Transforming frame {frame_idx} ({idx + 1}/{total})")
            
            async with semaphore:
                # Print when semaphore is acquired (should show concurrent execution)
                print(f"[ACQUIRED] Frame {frame_idx} acquired semaphore at {time.time():.2f}")
                
                try:
                    # Load mask for this frame if available
                    mask_path = masks_dir / f"mask_{frame_idx:04d}.png"
                    mask_path_param = mask_path if mask_path.exists() else None
                    
                    # Use reference frame for consistency
                    edited_bytes = await gemini_service.edit_frame_with_reference(
                        frame_path,
                        prompt,
                        preview_path,
                        mask_path=mask_path_param,
                    )

                    # Save transformed frame (resizes to match original if needed)
                    _save_edited_frame(frame_path, edited_bytes)
                    
                    elapsed = time.time() - start_time
                    print(f"[DONE] Frame {frame_idx} completed in {elapsed:.2f}s")
                    
                    completed_count += 1
                    project_manager.update_status(
                        project_id,
                        ai_edit_progress={"done": completed_count, "total": total},
                    )
                except Exception as e:
                    print(f"[ERROR] Error transforming frame {frame_idx}: {e}")
                    completed_count += 1
                    project_manager.update_status(
                        project_id,
                        ai_edit_progress={"done": completed_count, "total": total},
                    )
                    raise
        
        # Process all frames concurrently (limited by semaphore)
        print(f"[INFO] Starting concurrent transformation of {total} frames")
        tasks = [transform_frame(frame_idx, idx) for idx, frame_idx in enumerate(key_frames)]
        await asyncio.gather(*tasks)
        
        print(f"[INFO] AI transformation complete: transformed {total} key frames: {key_frames}")
        
        # Interpolate frames between key frames using RIFE (only within [start_frame, end_frame])
        print(f"[INFO] Starting RIFE interpolation between key frames")
        total_interpolation_frames = 0
        interpolation_segments = []
        for i in range(len(key_frames) - 1):
            start_frame_idx = key_frames[i]
            end_frame_idx = key_frames[i + 1]
            frames_to_interpolate = []
            for frame_idx in range(start_frame_idx + 1, end_frame_idx):
                if frame_idx < start_frame or frame_idx > end_frame:
                    continue
                frame_path = frames_dir / f"frame_{frame_idx:04d}.jpg"
                if frame_path.exists():
                    frames_to_interpolate.append(frame_path)
            if len(frames_to_interpolate) > 0:
                # Strict ascending order so RIFE writes the right blend to each frame file
                frames_to_interpolate.sort(key=lambda p: int(p.stem.replace("frame_", "")) if p.stem.startswith("frame_") else 0)
                total_interpolation_frames += len(frames_to_interpolate)
                interpolation_segments.append((start_frame_idx, end_frame_idx, frames_to_interpolate))
        
        if total_interpolation_frames > 0:
            project_manager.update_status(
                project_id,
                ai_edit_phase="interpolating",
                ai_interpolation_progress={"done": 0, "total": total_interpolation_frames},
            )
            
            async def interpolate_segment(start_frame_idx: int, end_frame_idx: int, frames_to_interpolate: list):
                """Interpolate frames between two key frames using RIFE."""
                if len(frames_to_interpolate) == 0:
                    return
                start_path = frames_dir / f"frame_{start_frame_idx:04d}.jpg"
                end_path = frames_dir / f"frame_{end_frame_idx:04d}.jpg"
                if not start_path.exists() or not end_path.exists():
                    print(f"[RIFE] Skipping interpolation: start or end frame missing")
                    return
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda sp=start_path, ep=end_path, fl=frames_to_interpolate: rife_service.interpolate_pair(sp, ep, fl),
                )
                print(f"[RIFE] Interpolated {len(frames_to_interpolate)} frames between {start_frame_idx} and {end_frame_idx}")
                status = project_manager.get_status(project_id)
                current_done = status.get("ai_interpolation_progress", {}).get("done", 0)
                project_manager.update_status(
                    project_id,
                    ai_interpolation_progress={"done": current_done + len(frames_to_interpolate), "total": total_interpolation_frames},
                )
            
            for start_frame_idx, end_frame_idx, frames_to_interpolate in interpolation_segments:
                await interpolate_segment(start_frame_idx, end_frame_idx, frames_to_interpolate)
            print(f"[INFO] RIFE interpolation complete: interpolated {total_interpolation_frames} frames")
        
        project_manager.update_status(
            project_id,
            ai_edit_status="done",
            ai_edit_phase="done",
            ai_edit_progress={"done": total, "total": total},
            ai_interpolation_progress={"done": total_interpolation_frames, "total": total_interpolation_frames},
            ai_edit_transformed_frames=key_frames,
            ai_generation_id=None,
        )
        print(f"AI edit complete: transformed {total} key frames, interpolated {total_interpolation_frames} in-between frames")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        project_manager.update_status(
            project_id,
            ai_edit_status="error",
            ai_edit_error=str(e),
        )


# --- Render ---

class RenderRequest(BaseModel):
    project_id: str

@app.post("/render")
async def render_video(req: RenderRequest):
    project_dir = project_manager.get_project_dir(req.project_id)
    frames_dir = project_dir / "frames"
    edited_dir = project_dir / "edited"
    output_path = project_dir / "output.mp4"

    status = project_manager.get_status(req.project_id)
    
    # Check if AI edits are done - use frames_dir, otherwise use edited_dir
    if status.get("ai_edit_status") == "done":
        # Use frames directory (contains AI-edited frames)
        ffmpeg_service.encode_video(frames_dir, output_path)
    else:
        # Check if regular edits are done
        if status.get("edit_status") not in ("done", None, "idle"):
            return {"error": f"Edit not complete. Current edit_status: {status.get('edit_status')}"}
        
        edited_frames = sorted(edited_dir.glob("frame_*.jpg"))
        if len(edited_frames) == 0:
            # No edits - use original frames
            ffmpeg_service.encode_video(frames_dir, output_path)
        else:
            ffmpeg_service.encode_video(edited_dir, output_path)

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
