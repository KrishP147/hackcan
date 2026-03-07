from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List
import shutil
import asyncio

from services import cloudinary_service, project_manager, ffmpeg_service, yolo_service, sam2_service

load_dotenv()
cloudinary_service.configure()

app = FastAPI(title="FrameShift AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


# --- Upload ---

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    project = project_manager.create_project()
    project_dir = project_manager.get_project_dir(project["project_id"])

    video_path = project_dir / "original.mp4"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = cloudinary_service.upload_file(str(video_path), resource_type="video")

    return {
        "project_id": project["project_id"],
        "video_url": result["url"],
        "public_id": result["public_id"],
    }


# --- Extract ---

class ExtractRequest(BaseModel):
    project_id: str

def _background_extract_and_detect(project_id: str):
    """Background task: extract frames and run YOLO on all."""
    project_dir = project_manager.get_project_dir(project_id)
    video_path = project_dir / "original.mp4"
    frames_dir = project_dir / "frames"

    # Extract frames
    project_manager.update_status(project_id, status="extracting")
    frame_count = ffmpeg_service.extract_frames(video_path, frames_dir)
    project_manager.update_status(project_id, status="detecting", frame_count=frame_count)

    # Run YOLO on all frames
    detections = {}
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    for i, f in enumerate(frame_files, start=1):
        dets = yolo_service.detect(f)
        detections[str(i)] = dets

    project_manager.update_status(project_id, status="ready", detections=detections)

@app.post("/extract")
async def extract_frames(req: ExtractRequest, background_tasks: BackgroundTasks):
    project_manager.update_status(req.project_id, status="processing")
    background_tasks.add_task(_background_extract_and_detect, req.project_id)
    return {"project_id": req.project_id, "status": "processing"}


@app.get("/project/{project_id}/status")
async def get_project_status(project_id: str):
    return project_manager.get_status(project_id)


@app.get("/frame/{project_id}/{frame_index}")
async def get_frame(project_id: str, frame_index: int):
    project_dir = project_manager.get_project_dir(project_id)
    frame_path = project_dir / "frames" / f"frame_{frame_index:04d}.jpg"
    if not frame_path.exists():
        return {"error": "Frame not found"}
    return FileResponse(frame_path, media_type="image/jpeg")


# --- Detect ---

class DetectRequest(BaseModel):
    project_id: str
    frame_index: int

@app.post("/detect")
async def detect_objects(req: DetectRequest):
    project_dir = project_manager.get_project_dir(req.project_id)
    frame_path = project_dir / "frames" / f"frame_{req.frame_index:04d}.jpg"

    if not frame_path.exists():
        return {"error": "Frame not found"}

    detections = yolo_service.detect(frame_path)
    return {"project_id": req.project_id, "frame_index": req.frame_index, "objects": detections}


# --- Segment ---

class SegmentRequest(BaseModel):
    project_id: str
    frame_index: int
    click_x: int
    click_y: int

@app.post("/segment")
async def segment_object(req: SegmentRequest):
    project_dir = project_manager.get_project_dir(req.project_id)
    frame_path = project_dir / "frames" / f"frame_{req.frame_index:04d}.jpg"
    masks_dir = project_dir / "masks"

    if not frame_path.exists():
        return {"error": "Frame not found"}

    mask = sam2_service.segment_frame(frame_path, req.click_x, req.click_y)
    mask_count = sam2_service.propagate_masks(
        project_dir / "original.mp4", req.frame_index, req.click_x, req.click_y, masks_dir
    )

    return {
        "project_id": req.project_id,
        "mask_count": mask_count,
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
    edit_type: str  # "recolor", "resize", "replace", "add", "delete"
    start_frame: int
    end_frame: int
    color: Optional[str] = None  # hex without #, e.g. "FF0000"
    scale: Optional[float] = None
    prompt: Optional[str] = None  # for replace and add
    asset_x: Optional[int] = None  # for add positioning
    asset_y: Optional[int] = None
    asset_w: Optional[int] = None
    asset_h: Optional[int] = None

class EditRequest(BaseModel):
    project_id: str
    edit_rules: List[EditRule]

async def _background_edit(project_id: str, edit_rules: List[EditRule]):
    """Background task: upload frames, apply edit rules, track progress in status.json."""
    try:
        project_dir = project_manager.get_project_dir(project_id)
        frames_dir = project_dir / "frames"
        masks_dir = project_dir / "masks"
        edited_dir = project_dir / "edited"
        edited_dir.mkdir(exist_ok=True)

        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        mask_files = sorted(masks_dir.glob("mask_*.png"))
        total_frames = len(frame_files)

        project_manager.update_status(
            project_id,
            edit_status="uploading",
            edit_progress={"done": 0, "total": total_frames},
        )

        # Upload all frames to Cloudinary
        frame_ids = {}
        for f in frame_files:
            result = cloudinary_service.upload_file(str(f), folder=f"frameshift/{project_id}/frames")
            frame_ids[f.name] = result["public_id"]

        # Upload masks if they exist
        mask_ids = {}
        for m in mask_files:
            result = cloudinary_service.upload_file(str(m), folder=f"frameshift/{project_id}/masks")
            mask_ids[m.name] = result["public_id"]

        # Compute mask bbox for positioning edits
        bbox_x, bbox_y, bbox_w, bbox_h = 0, 0, 0, 0
        if mask_files:
            from PIL import Image
            import numpy as np
            anchor_mask = np.array(Image.open(mask_files[0]))
            rows = np.any(anchor_mask > 0, axis=1)
            cols = np.any(anchor_mask > 0, axis=0)
            if rows.any() and cols.any():
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                bbox_x, bbox_y = int(x_min), int(y_min)
                bbox_w, bbox_h = int(x_max - x_min), int(y_max - y_min)

        frame_list = sorted(frame_ids.keys())
        mask_list = sorted(mask_ids.keys())

        frames_to_edit = set()
        for rule in edit_rules:
            for i in range(rule.start_frame, min(rule.end_frame + 1, total_frames + 1)):
                frames_to_edit.add(i)

        project_manager.update_status(project_id, edit_status="editing")
        completed = 0

        async def process_frame(index):
            nonlocal completed
            frame_name = frame_list[index - 1]
            f_id = frame_ids[frame_name]
            m_id = mask_ids.get(mask_list[index - 1], "") if index - 1 < len(mask_list) else ""

            if index not in frames_to_edit:
                shutil.copy2(str(frames_dir / frame_name), str(edited_dir / f"frame_{index:04d}.jpg"))
            else:
                url = None
                for rule in edit_rules:
                    if rule.start_frame <= index <= rule.end_frame:
                        if rule.edit_type == "recolor" and m_id:
                            url = await cloudinary_service.apply_recolor(f_id, m_id, rule.color)
                        elif rule.edit_type == "resize" and m_id:
                            url = await cloudinary_service.apply_resize(f_id, m_id, bbox_x, bbox_y, bbox_w, bbox_h, rule.scale)
                        elif rule.edit_type == "replace" and m_id:
                            url = await cloudinary_service.apply_replace(f_id, m_id, bbox_x, bbox_y, bbox_w, bbox_h)
                        elif rule.edit_type == "delete" and m_id:
                            url = await cloudinary_service.apply_delete(f_id, m_id)
                        elif rule.edit_type == "add":
                            url = await cloudinary_service.apply_add(
                                f_id, rule.prompt,
                                rule.asset_x or bbox_x, rule.asset_y or bbox_y,
                                rule.asset_w or bbox_w, rule.asset_h or bbox_h,
                            )
                        break

                save_path = edited_dir / f"frame_{index:04d}.jpg"
                if url:
                    await cloudinary_service.download_url(url, save_path)
                else:
                    shutil.copy2(str(frames_dir / frame_name), str(save_path))

            completed += 1
            project_manager.update_status(
                project_id,
                edit_progress={"done": completed, "total": total_frames},
            )

        batch_size = 20
        for i in range(0, total_frames, batch_size):
            batch = [process_frame(j + 1) for j in range(i, min(i + batch_size, total_frames))]
            await asyncio.gather(*batch)

        project_manager.update_status(
            project_id,
            edit_status="done",
            edit_progress={"done": total_frames, "total": total_frames},
        )

    except Exception as e:
        project_manager.update_status(project_id, edit_status="error", edit_error=str(e))


@app.post("/edit")
async def edit_frames(req: EditRequest, background_tasks: BackgroundTasks):
    project_dir = project_manager.get_project_dir(req.project_id)
    if not any((project_dir / "frames").glob("frame_*.jpg")):
        return {"error": "No frames found. Run /extract first."}

    project_manager.update_status(
        req.project_id,
        edit_status="uploading",
        edit_progress={"done": 0, "total": 0},
    )
    background_tasks.add_task(_background_edit, req.project_id, req.edit_rules)
    return {"project_id": req.project_id, "edit_status": "uploading"}


# --- Render ---

class RenderRequest(BaseModel):
    project_id: str

@app.post("/render")
async def render_video(req: RenderRequest):
    project_dir = project_manager.get_project_dir(req.project_id)
    edited_dir = project_dir / "edited"
    output_path = project_dir / "output.mp4"

    status = project_manager.get_status(req.project_id)
    if status.get("edit_status") not in ("done", None, "idle"):
        return {"error": f"Edit not complete. Current edit_status: {status.get('edit_status')}"}

    edited_frames = sorted(edited_dir.glob("frame_*.jpg"))
    if len(edited_frames) == 0:
        return {"error": "No edited frames found. Run /edit first."}

    ffmpeg_service.encode_video(edited_dir, output_path)

    result = cloudinary_service.upload_file(str(output_path), resource_type="video")

    return {
        "project_id": req.project_id,
        "video_url": result["url"],
        "cloudinary_public_id": result["public_id"],
    }
