from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List
import shutil
import asyncio
import numpy as np

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

def _background_extract(project_id: str):
    """Background task: extract frames, then kick off YOLO in background."""
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

    # Mark ready immediately so frontend can show frames
    project_manager.update_status(project_id, status="ready", frame_count=frame_count,
                                   frame_width=frame_width, frame_height=frame_height,
                                   detecting=True, detections={})

    # Run YOLO on all frames, updating detections progressively
    detections = {}
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    for i, f in enumerate(frame_files, start=1):
        dets = yolo_service.detect(f)
        detections[str(i)] = dets
        # Update every 10 frames so frontend can poll partial results
        if i % 10 == 0 or i == len(frame_files):
            project_manager.update_status(project_id, detections=detections, detected_frames=i)

    project_manager.update_status(project_id, detecting=False, detections=detections, detected_frames=len(frame_files))

@app.post("/extract")
async def extract_frames(req: ExtractRequest, background_tasks: BackgroundTasks):
    project_manager.update_status(req.project_id, status="processing")
    background_tasks.add_task(_background_extract, req.project_id)
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

def _background_segment_and_propagate(project_id: str, frame_index: int, click_x: int, click_y: int):
    """Background task: segment the single clicked frame only."""
    project_dir = project_manager.get_project_dir(project_id)
    frame_path = project_dir / "frames" / f"frame_{frame_index:04d}.jpg"
    masks_dir = project_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    project_manager.update_status(project_id, segmenting=True, segment_status="segmenting")

    mask = sam2_service.segment_frame(frame_path, click_x, click_y)

    # Save mask for just this frame
    from PIL import Image
    mask_img = (mask.astype(np.uint8)) * 255
    mask_path = masks_dir / f"mask_{frame_index:04d}.png"
    Image.fromarray(mask_img).save(mask_path)

    project_manager.update_status(
        project_id, segmenting=False, segment_status="done",
        mask_count=1, anchor_frame=frame_index,
    )

@app.post("/segment")
async def segment_object(req: SegmentRequest, background_tasks: BackgroundTasks):
    project_dir = project_manager.get_project_dir(req.project_id)
    frame_path = project_dir / "frames" / f"frame_{req.frame_index:04d}.jpg"

    if not frame_path.exists():
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
    """Apply color tint locally using SAM2 mask — no Cloudinary needed."""
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


# Edits that should be masked locally (applied to full frame by Cloudinary, then composited)
MASK_EDITS = {"delete", "replace", "resize", "blur_region", "gen_recolor"}
# Edits done entirely locally with mask
LOCAL_EDITS = {"recolor"}
# Edits that affect the whole frame (no masking)
FRAME_EDITS = {"bg_remove", "bg_replace", "gen_fill", "enhance", "upscale", "restore", "blur", "drop_shadow"}


async def _background_edit(project_id: str, edit_rules: List[EditRule]):
    """Background task: upload only the needed frames, apply edits, download results."""
    try:
        from PIL import Image

        project_dir = project_manager.get_project_dir(project_id)
        frames_dir = project_dir / "frames"
        masks_dir = project_dir / "masks"

        # Collect unique frame indices to edit
        frames_to_edit: set[int] = set()
        for rule in edit_rules:
            for i in range(rule.start_frame, rule.end_frame + 1):
                frames_to_edit.add(i)

        total = len(frames_to_edit)
        project_manager.update_status(project_id, edit_status="uploading", edit_progress={"done": 0, "total": total})

        # Load SAM2 mask if available
        mask_array = None
        mask_files = sorted(masks_dir.glob("mask_*.png"))
        bbox_x, bbox_y, bbox_w, bbox_h = 0, 0, 0, 0
        if mask_files:
            mask_array = np.array(Image.open(mask_files[0]))
            rows = np.any(mask_array > 0, axis=1)
            cols = np.any(mask_array > 0, axis=0)
            if rows.any() and cols.any():
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                bbox_x, bbox_y = int(x_min), int(y_min)
                bbox_w, bbox_h = int(x_max - x_min), int(y_max - y_min)

        # Check if any rule needs Cloudinary (not purely local)
        needs_cloudinary = any(r.edit_type not in LOCAL_EDITS for r in edit_rules)

        # Upload frames to Cloudinary only if needed
        frame_ids: dict[int, str] = {}
        if needs_cloudinary:
            for idx in sorted(frames_to_edit):
                frame_path = frames_dir / f"frame_{idx:04d}.jpg"
                if not frame_path.exists():
                    continue
                result = cloudinary_service.upload_file(str(frame_path), folder=f"frameshift/{project_id}/frames")
                frame_ids[idx] = result["public_id"]

        project_manager.update_status(project_id, edit_status="editing")
        completed = 0

        for idx in sorted(frames_to_edit):
            frame_path = frames_dir / f"frame_{idx:04d}.jpg"
            if not frame_path.exists():
                completed += 1
                continue

            f_id = frame_ids.get(idx, "")

            for rule in edit_rules:
                if rule.start_frame <= idx <= rule.end_frame:
                    t = rule.edit_type

                    # ── Local-only edits (use mask directly with PIL) ──
                    if t == "recolor" and mask_array is not None:
                        _apply_recolor_local(frame_path, mask_array, rule.color or "FF0000")

                    # ── Cloudinary edits that need local mask compositing ──
                    elif t in MASK_EDITS and f_id:
                        url = None
                        if t == "delete":
                            url = await cloudinary_service.apply_delete(f_id, "")
                        elif t == "replace":
                            url = await cloudinary_service.apply_replace(f_id, "", rule.prompt or "object")
                        elif t == "resize":
                            url = await cloudinary_service.apply_resize(f_id, "", bbox_x, bbox_y, bbox_w, bbox_h, rule.scale or 1.5)
                        elif t == "blur_region":
                            # Apply blur to full frame, then composite
                            url = await cloudinary_service.apply_blur(f_id, 1000)
                        elif t == "gen_recolor":
                            url = await cloudinary_service.apply_generative_recolor(f_id, rule.prompt or "object", rule.color or "FF0000")

                        if url and mask_array is not None:
                            tmp_path = frame_path.with_suffix(".edited.jpg")
                            await cloudinary_service.download_url(url, tmp_path)
                            _composite_with_mask(frame_path, tmp_path, mask_array)
                            # Move composited result to original location
                            shutil.move(str(tmp_path), str(frame_path))

                    # ── Whole-frame edits (no mask needed) ──
                    elif t in FRAME_EDITS and f_id:
                        url = None
                        if t == "bg_remove":
                            url = await cloudinary_service.apply_background_remove(f_id)
                        elif t == "bg_replace":
                            url = await cloudinary_service.apply_background_replace(f_id, rule.prompt or "studio background")
                        elif t == "gen_fill":
                            url = await cloudinary_service.apply_generative_fill(f_id, rule.prompt)
                        elif t == "enhance":
                            url = await cloudinary_service.apply_enhance(f_id)
                        elif t == "upscale":
                            url = await cloudinary_service.apply_upscale(f_id)
                        elif t == "restore":
                            url = await cloudinary_service.apply_restore(f_id)
                        elif t == "blur":
                            url = await cloudinary_service.apply_blur(f_id, rule.blur_strength or 500)
                        elif t == "drop_shadow":
                            url = await cloudinary_service.apply_drop_shadow(f_id)

                        if url:
                            await cloudinary_service.download_url(url, frame_path)

                    break

            completed += 1
            project_manager.update_status(project_id, edit_progress={"done": completed, "total": total})

        project_manager.update_status(project_id, edit_status="done", edit_progress={"done": total, "total": total})

    except Exception as e:
        import traceback
        traceback.print_exc()
        project_manager.update_status(project_id, edit_status="error", edit_error=str(e))


@app.post("/edit")
async def edit_frames(req: EditRequest):
    project_dir = project_manager.get_project_dir(req.project_id)
    if not any((project_dir / "frames").glob("frame_*.jpg")):
        return {"error": "No frames found. Run /extract first."}

    project_manager.update_status(
        req.project_id,
        edit_status="uploading",
        edit_progress={"done": 0, "total": 0},
    )
    # Run as a proper async task instead of BackgroundTasks (which can't await)
    asyncio.ensure_future(_background_edit(req.project_id, req.edit_rules))
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
