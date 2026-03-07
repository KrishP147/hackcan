from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
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

@app.post("/extract")
async def extract_frames(req: ExtractRequest):
    project_dir = project_manager.get_project_dir(req.project_id)
    video_path = project_dir / "original.mp4"
    frames_dir = project_dir / "frames"

    frame_count = ffmpeg_service.extract_frames(video_path, frames_dir)

    return {"project_id": req.project_id, "frame_count": frame_count}


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
        project_dir / "frames", req.frame_index, mask, masks_dir
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

class EditRequest(BaseModel):
    project_id: str
    edit_type: str  # "recolor", "resize", "replace"
    color: Optional[str] = None  # hex without #, e.g. "FF0000"
    scale: Optional[float] = None
    replacement_public_id: Optional[str] = None

@app.post("/edit")
async def edit_frames(req: EditRequest):
    project_dir = project_manager.get_project_dir(req.project_id)
    frames_dir = project_dir / "frames"
    masks_dir = project_dir / "masks"
    edited_dir = project_dir / "edited"
    edited_dir.mkdir(exist_ok=True)

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    mask_files = sorted(masks_dir.glob("mask_*.png"))

    if len(frame_files) == 0 or len(mask_files) == 0:
        return {"error": "No frames or masks found. Run /extract and /segment first."}

    # Upload all frames and masks to Cloudinary
    frame_ids = {}
    mask_ids = {}

    for f in frame_files:
        result = cloudinary_service.upload_file(str(f), folder=f"frameshift/{req.project_id}/frames")
        frame_ids[f.name] = result["public_id"]

    for m in mask_files:
        result = cloudinary_service.upload_file(str(m), folder=f"frameshift/{req.project_id}/masks")
        mask_ids[m.name] = result["public_id"]

    # Get mask bbox from first mask for positioning
    from PIL import Image
    import numpy as np
    anchor_mask = np.array(Image.open(mask_files[0]))
    rows = np.any(anchor_mask > 0, axis=1)
    cols = np.any(anchor_mask > 0, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    bbox_x, bbox_y = int(x_min), int(y_min)
    bbox_w, bbox_h = int(x_max - x_min), int(y_max - y_min)

    # Apply transforms and download results
    async def process_frame(frame_name, mask_name, index):
        f_id = frame_ids[frame_name]
        m_id = mask_ids[mask_name]

        if req.edit_type == "recolor":
            url = await cloudinary_service.apply_recolor(f_id, m_id, req.color)
        elif req.edit_type == "resize":
            url = await cloudinary_service.apply_resize(f_id, m_id, bbox_x, bbox_y, bbox_w, bbox_h, req.scale)
        elif req.edit_type == "replace":
            url = await cloudinary_service.apply_replace(f_id, req.replacement_public_id, bbox_x, bbox_y, bbox_w, bbox_h)
        else:
            return

        save_path = edited_dir / f"frame_{index:04d}.jpg"
        await cloudinary_service.download_url(url, save_path)

    # Process in batches of 20 concurrent
    batch_size = 20
    frame_list = sorted(frame_ids.keys())
    mask_list = sorted(mask_ids.keys())

    for i in range(0, len(frame_list), batch_size):
        batch = []
        for j in range(i, min(i + batch_size, len(frame_list))):
            batch.append(process_frame(frame_list[j], mask_list[j], j + 1))
        await asyncio.gather(*batch)

    return {"project_id": req.project_id, "edited_frame_count": len(frame_list), "status": "done"}
