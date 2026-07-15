from fastapi import (FastAPI, UploadFile, File, BackgroundTasks, Depends,
                     HTTPException, Header)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# Load local configuration before importing services that read environment
# variables (notably Auth0 and the projects directory).
load_dotenv()

from pydantic import BaseModel
from typing import Optional, List
import shutil
import asyncio
import math
from functools import partial
import numpy as np
import uuid
import json
import os
import re
import secrets
import tarfile
import tempfile
from pathlib import Path
from io import BytesIO
from urllib.parse import urlparse

import httpx

from services import (project_manager, ffmpeg_service, sam2_service, config,
                      supabase_storage_service)
from services.auth_service import get_current_user, is_auth0_configured
# film_service  # FILM disabled - using RIFE instead

app = FastAPI(title="FrameShift AI")

# Track cancellable operations per project
_cancel_flags: dict[str, bool] = {}

import os as _os_cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in ("http://localhost:3000",
                               _os_cors.getenv("FRONTEND_ORIGIN")) if o],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization"],
)

# Per-IP token bucket (30 req/min, refill 0.5/s) — enabled via RATE_LIMIT=1 (Modal deploy)
import os as _os
import time as _time

_rate_buckets: dict[str, tuple[float, float]] = {}  # ip -> (tokens, last_refill)

if _os.getenv("RATE_LIMIT", "0") == "1":
    @app.middleware("http")
    async def rate_limit(request, call_next):
        from fastapi.responses import JSONResponse
        ip = request.client.host if request.client else "unknown"
        tokens, last = _rate_buckets.get(ip, (30.0, _time.monotonic()))
        now = _time.monotonic()
        tokens = min(30.0, tokens + (now - last) * 0.5)
        if tokens < 1.0:
            _rate_buckets[ip] = (tokens, now)
            return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)
        _rate_buckets[ip] = (tokens - 1.0, now)
        return await call_next(request)


@app.on_event("startup")
async def startup():
    project_manager.reset_stuck_projects()
    print(f"[Startup] compute device: {config.get_device()}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "compute_device": str(config.get_device()),
        "auth0_configured": is_auth0_configured(),
    }


# --- Upload ---

MAX_UPLOAD_DURATION_SECONDS = 6.0
PROJECT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def _validate_upload_duration(video_path: Path) -> float:
    try:
        duration = float(ffmpeg_service.probe_video(video_path)["duration"])
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid, supported video file.",
        ) from exc

    if not math.isfinite(duration) or duration <= 0:
        raise HTTPException(
            status_code=400,
            detail="Could not determine the video's duration.",
        )
    if duration >= MAX_UPLOAD_DURATION_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Video must be under {MAX_UPLOAD_DURATION_SECONDS:g} seconds "
                f"(received {duration:.1f} seconds)."
            ),
        )

    return duration


class ImportProjectRequest(BaseModel):
    project_id: str
    source_url: str
    checkpoint_url: str | None = None
    user_id: str | None = None
    original_path: str | None = None
    current_path: str | None = None
    checkpoint_path: str | None = None


def _validate_storage_url(value: str) -> None:
    expected_host = urlparse(os.getenv("SUPABASE_URL", "")).hostname
    parsed = urlparse(value)
    if (not expected_host or parsed.scheme != "https" or
            parsed.hostname != expected_host or
            not parsed.path.startswith("/storage/v1/object/sign/")):
        raise ValueError("Only signed URLs from the configured Supabase project are allowed")


def _download_signed_object(url: str, destination: Path) -> None:
    _validate_storage_url(url)
    temporary = destination.with_suffix(destination.suffix + ".download")
    with httpx.stream("GET", url, follow_redirects=True, timeout=180.0) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_bytes():
                output.write(chunk)
    temporary.replace(destination)


def _restore_checkpoint(project_id: str, checkpoint_url: str) -> None:
    project_dir = project_manager.get_project_dir(project_id)
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "checkpoint.tar.gz"
        _download_signed_object(checkpoint_url, archive_path)
        with tarfile.open(archive_path, "r:gz") as archive:
            status_member = archive.getmember("status.json") if "status.json" in archive.getnames() else None
            if status_member:
                status_file = archive.extractfile(status_member)
                if status_file:
                    saved = json.load(status_file)
                    restorable = {
                        key: saved[key]
                        for key in (
                            "anchor_frame", "click_x", "click_y", "mask_count",
                            "segment_status", "edit_version", "flow_status",
                        )
                        if key in saved
                    }
                    project_manager.update_status(project_id, **restorable)

            masks_dir = project_dir / "masks"
            masks_dir.mkdir(exist_ok=True)
            for member in archive.getmembers():
                if not member.isfile() or not re.fullmatch(r"masks/mask_\d{4}\.png", member.name):
                    continue
                source = archive.extractfile(member)
                if source:
                    (masks_dir / Path(member.name).name).write_bytes(source.read())


def _background_import_project(req: ImportProjectRequest) -> None:
    try:
        project = project_manager.create_project(req.project_id)
        project_dir = Path(project["project_dir"])
        project_manager.update_status(
            req.project_id,
            status="hydrating",
            user_id=req.user_id,
            storage_original_path=req.original_path,
            storage_current_path=req.current_path,
            storage_checkpoint_path=req.checkpoint_path,
            error=None,
        )
        video_path = project_dir / "original.mp4"
        _download_signed_object(req.source_url, video_path)
        duration = _validate_upload_duration(video_path)
        project_manager.update_status(req.project_id, duration=duration)

        if req.checkpoint_url:
            _restore_checkpoint(req.project_id, req.checkpoint_url)

        # A hydrated cache is immediately made editor-ready. The browser does
        # not need to issue a second upload or extraction request.
        _background_extract(req.project_id)
    except Exception as exc:
        print(f"[import] project hydration failed for {req.project_id}: {exc}")
        try:
            project_manager.update_status(
                req.project_id, status="error", error=f"Storage hydration failed: {exc}")
        except Exception:
            pass


@app.post("/project/import", status_code=202)
async def import_project_from_storage(
    req: ImportProjectRequest,
    background_tasks: BackgroundTasks,
    x_frameshift_import_secret: str | None = Header(default=None),
):
    configured_secret = os.getenv("FRAMESHIFT_IMPORT_SECRET", "")
    if not configured_secret:
        raise HTTPException(status_code=503, detail="Storage hydration is not configured")
    if not x_frameshift_import_secret or not secrets.compare_digest(
        x_frameshift_import_secret, configured_secret
    ):
        raise HTTPException(status_code=401, detail="Invalid storage hydration secret")
    if not PROJECT_ID_PATTERN.fullmatch(req.project_id):
        raise HTTPException(status_code=400, detail="Invalid project_id")
    try:
        _validate_storage_url(req.source_url)
        if req.checkpoint_url:
            _validate_storage_url(req.checkpoint_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    existing_status = project_manager.get_status(req.project_id)
    try:
        existing_dir = project_manager.get_project_dir(req.project_id)
        if (existing_status.get("status") == "ready" and
                any((existing_dir / "frames").glob("frame_*.jpg"))):
            return {"project_id": req.project_id, "status": "ready", "cache": "hit"}
    except FileNotFoundError:
        pass

    project_manager.create_project(req.project_id)
    project_manager.update_status(req.project_id, status="hydrating", error=None)
    background_tasks.add_task(_background_import_project, req)
    return {"project_id": req.project_id, "status": "hydrating", "cache": "miss"}

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
    try:
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        duration = _validate_upload_duration(video_path)
    except Exception:
        shutil.rmtree(project_dir, ignore_errors=True)
        raise

    project_manager.update_status(project["project_id"], duration=duration)

    # Video is staged on the local project volume for FFmpeg processing.
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
        if info["duration"] >= MAX_UPLOAD_DURATION_SECONDS:
            project_manager.update_status(
                project_id, status="error",
                error="Video must be under 6 seconds")
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

    # Durable media lives in Supabase. The Volume remains only a fast cache.
    supabase_storage_service.sync_extract_metadata(project_id)

    # Flow is lazy by default. Starting RAFT here used to contend with SAM2 as
    # soon as the user clicked an object. It can still be enabled explicitly
    # for local benchmarking.
    if config.should_precompute_flows():
        import threading
        threading.Thread(target=_background_flows, args=(project_id,), daemon=True).start()
    else:
        project_manager.update_status(project_id, flow_status="deferred")

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

def _background_flows(project_id: str):
    """Pairwise RAFT flow over the whole clip, cached to flows/. Safe to call
    repeatedly — already-cached pairs are skipped."""
    try:
        status = project_manager.get_status(project_id)
        if status.get("flow_status") == "computing":
            return
        project_manager.update_status(project_id, flow_status="computing")
        from services import config as _config, flow_service
        project_dir = project_manager.get_project_dir(project_id)
        with _config.gpu_job(f"raft:{project_id}"):
            flow_service.compute_flows(
                project_dir / "frames", project_dir / "flows",
                device=_config.get_device(),
            )
        project_manager.update_status(project_id, flow_status="done")
    except Exception as e:
        project_manager.update_status(project_id, flow_status="error", flow_error=str(e))


@app.post("/flows/precompute")
async def precompute_flows(req: ExtractRequest):
    """Manually invoke optical-flow precompute for a project (idempotent)."""
    asyncio.ensure_future(asyncio.to_thread(_background_flows, req.project_id))
    return {"project_id": req.project_id, "flow_status": "computing"}


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


def _segment_frame_serialized(frame_path: Path, click_x: int, click_y: int):
    with config.gpu_job("sam2:keyframe"):
        return sam2_service.segment_frame(frame_path, click_x, click_y)


def _propagate_masks_serialized(propagate):
    with config.gpu_job("sam2:tracking"):
        return propagate()


async def _background_segment_keyframe(
    project_id: str, frame_index: int, click_x: int, click_y: int
):
    """Segment only the selected keyframe and wait for user confirmation."""
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

        project_manager.update_status(
            project_id,
            segmenting=True,
            segment_status="segmenting",
            segment_error=None,
        )

        print(f"[SAM2] Starting segmentation for frame {frame_index} at ({click_x}, {click_y})")
        
        # Segment only the clicked frame (no propagation)
        loop = asyncio.get_event_loop()
        mask = await loop.run_in_executor(
            None,
            _segment_frame_serialized,
            frame_path,
            click_x,
            click_y
        )
        print(f"[SAM2] Segmentation complete, mask shape: {mask.shape}")

        # Save mask for this frame only
        from PIL import Image
        # A new click starts a new object track. Do not let masks from a
        # previous selection make the next edit skip propagation.
        for old_mask in masks_dir.glob("mask_*.png"):
            old_mask.unlink()
        mask_img = (mask.astype(np.uint8)) * 255
        mask_path = masks_dir / f"mask_{frame_index:04d}.png"
        Image.fromarray(mask_img).save(mask_path)
        from services import mask_service
        mask_service.condition_single(mask_path)
        print(f"[SAM2] Saved mask to {mask_path}")

        project_manager.update_status(
            project_id, segmenting=False, segment_status="keyframe_ready",
            mask_count=1, anchor_frame=frame_index,
            click_x=click_x, click_y=click_y,
            segment_error=None,
        )
        print(f"[SAM2] Keyframe mask ready; awaiting propagation confirmation")
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[SAM2] Error during keyframe segmentation: {str(e)}")
        print(error_trace)
        project_manager.update_status(
            project_id,
            segmenting=False,
            segment_status="error",
            segment_error=str(e),
        )


async def _background_propagate_segment(project_id: str):
    """Track a confirmed keyframe mask through the complete clip."""
    try:
        project_dir = project_manager.get_project_dir(project_id)
        frames_dir = project_dir / "frames"
        masks_dir = project_dir / "masks"
        status = project_manager.get_status(project_id)
        frame_index = status.get("anchor_frame")
        click_x = status.get("click_x")
        click_y = status.get("click_y")
        if not frame_index:
            raise RuntimeError("No keyframe selection is ready")

        mask_path = masks_dir / f"mask_{frame_index:04d}.png"
        if not mask_path.exists():
            raise RuntimeError("The keyframe mask is missing; select the object again")

        from PIL import Image
        from services import mask_service

        project_manager.update_status(
            project_id,
            segmenting=True,
            segment_status="propagating",
            segment_error=None,
        )
        loop = asyncio.get_running_loop()
        propagate = partial(
            sam2_service.propagate_masks,
            frames_dir,
            frame_index,
            (np.array(Image.open(mask_path).convert("L")) > 128),
            masks_dir,
            click_x=click_x,
            click_y=click_y,
            frame_step=config.get_mask_frame_step(),
            cancel_check=lambda: _cancel_flags.get(project_id, False),
        )
        mask_count = await loop.run_in_executor(
            None, _propagate_masks_serialized, propagate)
        mask_service.stabilize_masks(masks_dir)
        project_manager.update_status(
            project_id,
            segmenting=False,
            segment_status="done",
            mask_count=mask_count,
            segment_error=None,
        )
        await asyncio.to_thread(supabase_storage_service.sync_checkpoint, project_id)
        print(f"[SAM2] Full mask propagation complete: {mask_count} frames")
    except sam2_service.PropagationCancelled:
        project_manager.update_status(
            project_id, segmenting=False, segment_status="cancelled")
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
    """Segment an object on one keyframe; propagation requires confirmation."""
    print(f"[SAM2] /segment endpoint called: project={req.project_id}, frame={req.frame_index}, click=({req.click_x}, {req.click_y})")
    
    project_dir = project_manager.get_project_dir(req.project_id)
    frame_path = project_dir / "frames" / f"frame_{req.frame_index:04d}.jpg"

    current_status = project_manager.get_status(req.project_id)
    if current_status.get("segment_status") in {"segmenting", "propagating"} or current_status.get("segmenting"):
        return {"error": "Segmentation is already in progress."}
    if current_status.get("edit_status") in {"uploading", "editing", "processing"}:
        return {"error": "Wait for the current edit to finish before segmenting again."}

    if not frame_path.exists():
        print(f"[SAM2] Error: Frame not found at {frame_path}")
        return {"error": "Frame not found"}

    # Mark the new request before scheduling the background task. This prevents
    # polling from briefly treating a second click as the previous completed mask.
    project_manager.update_status(
        req.project_id,
        segmenting=True,
        segment_status="segmenting",
        segment_error=None,
    )
    _cancel_flags[req.project_id] = False

    background_tasks.add_task(
        _background_segment_keyframe,
        req.project_id, req.frame_index, req.click_x, req.click_y,
    )

    return {
        "project_id": req.project_id,
        "status": "processing",
        "anchor_frame": req.frame_index,
    }


class SegmentPropagationRequest(BaseModel):
    project_id: str


@app.post("/segment/propagate")
async def propagate_segment(
    req: SegmentPropagationRequest, background_tasks: BackgroundTasks
):
    """Propagate a prepared keyframe mask after explicit user confirmation."""
    status = project_manager.get_status(req.project_id)
    if status.get("segment_status") in {"segmenting", "propagating"} or status.get("segmenting"):
        return {"error": "Segmentation is already in progress."}
    if status.get("segment_status") != "keyframe_ready":
        return {"error": "Select an object on a keyframe before tracking it."}
    if status.get("edit_status") in {"uploading", "editing", "processing"}:
        return {"error": "Wait for the current edit to finish before tracking the mask."}

    _cancel_flags[req.project_id] = False
    project_manager.update_status(
        req.project_id,
        segmenting=True,
        segment_status="propagating",
        segment_error=None,
    )
    background_tasks.add_task(_background_propagate_segment, req.project_id)
    return {"project_id": req.project_id, "status": "propagating"}


@app.get("/mask/{project_id}/{mask_index}")
async def get_mask(project_id: str, mask_index: int):
    project_dir = project_manager.get_project_dir(project_id)
    mask_path = project_dir / "masks" / f"mask_{mask_index:04d}.png"
    if not mask_path.exists():
        return {"error": "Mask not found"}
    return FileResponse(mask_path, media_type="image/png")


@app.get("/mask-outline/{project_id}/{mask_index}")
async def get_mask_outline(project_id: str, mask_index: int):
    """Transparent accent contour for responsive editor playback."""
    from fastapi.responses import Response
    import cv2

    project_dir = project_manager.get_project_dir(project_id)
    mask_path = project_dir / "masks" / f"mask_{mask_index:04d}.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise HTTPException(status_code=404, detail="Mask not found")

    hard = (mask > 128).astype(np.uint8) * 255
    edge = cv2.morphologyEx(
        hard,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    )
    rgba = np.zeros((*hard.shape, 4), dtype=np.uint8)
    # OpenCV encodes BGRA; the FrameShift accent is RGB(244, 63, 94).
    rgba[edge > 0] = (94, 63, 244, 255)
    ok, encoded = cv2.imencode(".png", rgba)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not render mask outline")
    return Response(
        content=encoded.tobytes(),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


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
    preview_frame: Optional[int] = None  # frame the user is viewing — edit lands
                                         # there first, then sweeps outward

class EditRequest(BaseModel):
    project_id: str
    edit_rules: List[EditRule]


def _prepared_preview_generator(project_id: str, rule: EditRule):
    """Return a generate-once wrapper for an approved generative preview."""
    if rule.edit_type not in {"replace", "bg_replace"}:
        return None
    status = project_manager.get_status(project_id)
    pending = status.get("pending_edit_preview") or {}
    preview_frame = rule.preview_frame or rule.start_frame
    if (
        pending.get("edit_type") != rule.edit_type
        or pending.get("frame_index") != preview_frame
        or pending.get("prompt", "") != (rule.prompt or "")
    ):
        return None

    preview_path = project_manager.get_project_dir(project_id) / "pending" / "edit_anchor.jpg"
    if not preview_path.exists():
        return None

    if rule.edit_type == "replace":
        from services import replace_tool
        fallback = replace_tool._default_generate
    else:
        from services import background_tool
        fallback = background_tool._default_generate
    used = False

    async def generate(frame_path, prompt, reference_frame_path=None, mask_path=None):
        nonlocal used
        if not used and reference_frame_path is None:
            used = True
            return preview_path.read_bytes()
        return await fallback(
            frame_path,
            prompt,
            reference_frame_path=reference_frame_path,
            mask_path=mask_path,
        )

    return generate



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

        total_frames = sum(
            max(0, rule.end_frame - rule.start_frame + 1)
            for rule in edit_rules
        )
        project_manager.update_status(
            project_id,
            last_backup_timestamp=backup_timestamp,
            last_backup_frames=sorted(frames_to_edit),
            edit_status="editing",
            edit_error=None,
            edit_phase="tracking",
            edit_progress={"done": 0, "total": total_frames},
            edit_sweep=None,
        )

        from services import edit_dispatch
        completed_frames = 0
        for rule in edit_rules:
            if _cancel_flags.get(project_id):
                project_manager.update_status(project_id, edit_status="cancelled")
                return

            rule_total = max(0, rule.end_frame - rule.start_frame + 1)

            def update_progress(done: int, _total: int, base=completed_frames):
                project_manager.update_status(
                    project_id,
                    edit_progress={
                        "done": min(base + done, total_frames),
                        "total": total_frames,
                    },
                )

            prepared_generate = _prepared_preview_generator(project_id, rule)
            try:
                await edit_dispatch.run_edit_rule(
                    project_id, rule,
                    progress_cb=update_progress,
                    cancel_check=lambda: _cancel_flags.get(project_id, False),
                    generate=prepared_generate,
                )
            finally:
                if prepared_generate is not None:
                    pending_path = project_dir / "pending" / "edit_anchor.jpg"
                    pending_path.unlink(missing_ok=True)
                    project_manager.update_status(project_id, pending_edit_preview=None)
            if _cancel_flags.get(project_id):
                project_manager.update_status(project_id, edit_status="cancelled")
                return
            project_manager.update_status(project_id, edit_phase="editing")
            completed_frames += rule_total
            project_manager.update_status(
                project_id,
                edit_progress={"done": completed_frames, "total": total_frames},
            )

        project_manager.update_status(
            project_id, edit_status="done",
            edit_phase="done",
            edit_progress={"done": total_frames, "total": total_frames},
            edit_version=(project_manager.get_status(project_id).get("edit_version", 0) or 0) + 1,
        )
        await asyncio.to_thread(supabase_storage_service.sync_current_video, project_id)
    except Exception as e:
        from services import edit_dispatch, sam2_service
        if isinstance(e, (edit_dispatch.EditCancelled, sam2_service.PropagationCancelled)):
            project_manager.update_status(project_id, edit_status="cancelled")
            return
        import traceback
        traceback.print_exc()
        project_manager.update_status(
            project_id, edit_status="error", edit_phase="error", edit_error=str(e))


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
    await asyncio.to_thread(supabase_storage_service.sync_current_video, req.project_id)
    
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
    if not req.edit_rules:
        return {"error": "No edit rules provided."}

    current_status = project_manager.get_status(req.project_id)
    if current_status.get("segment_status") in {"segmenting", "propagating"} or current_status.get("segmenting"):
        return {"error": "Wait for segmentation to finish before applying Blur."}
    if current_status.get("edit_status") in {"uploading", "editing", "processing"}:
        return {"error": "An edit is already being applied."}

    project_manager.update_status(
        req.project_id,
        edit_status="editing",
        edit_error=None,
        edit_phase="tracking",
        edit_progress={"done": 0, "total": 0},
        edit_sweep=None,
    )
    # Run as a proper async task instead of BackgroundTasks (which can't await)
    asyncio.ensure_future(_background_edit(req.project_id, req.edit_rules))
    return {"project_id": req.project_id, "edit_status": "editing"}


class PreviewRequest(BaseModel):
    project_id: str
    frame_index: int
    edit_type: str
    color: Optional[str] = None
    blur_strength: Optional[int] = None
    scale: Optional[float] = None
    prompt: Optional[str] = None
    dx: Optional[int] = None
    dy: Optional[int] = None


@app.post("/edit/preview")
async def edit_preview(req: PreviewRequest):
    """Instant single-frame preview: JPEG of the edit applied to one frame,
    in memory. The durable propagation runs separately via /edit."""
    from fastapi.responses import Response
    from services import preview_service
    try:
        project_dir = project_manager.get_project_dir(req.project_id)
        pending_dir = project_dir / "pending"
        pending_path = pending_dir / "edit_anchor.jpg"
        if req.edit_type in {"replace", "bg_replace"}:
            frame_path = project_dir / "frames" / f"frame_{req.frame_index:04d}.jpg"
            mask_path = project_dir / "masks" / f"mask_{req.frame_index:04d}.png"
            if not frame_path.exists():
                raise FileNotFoundError(f"Frame {req.frame_index} not found")
            if not mask_path.exists():
                raise RuntimeError("No mask — click an object first")
            from services import gemini_service
            validation = await gemini_service.validate_edit_prompt(
                frame_path,
                mask_path,
                req.edit_type,
                req.prompt or "",
            )
            if not validation.valid:
                raise ValueError(
                    f"Try again with a better prompt. {validation.reason}"
                )
            if req.edit_type == "replace":
                from services import replace_tool
                generated = await replace_tool._default_generate(
                    frame_path, req.prompt or "", mask_path=mask_path)
                jpeg = generated
            else:
                from services import background_tool, mask_service
                import cv2
                generated = await background_tool._default_generate(
                    frame_path, req.prompt or "", mask_path=mask_path)
                frame = cv2.imread(str(frame_path))
                plate = cv2.imdecode(np.frombuffer(generated, np.uint8), cv2.IMREAD_COLOR)
                if plate is None:
                    raise RuntimeError("Generated background could not be decoded")
                if plate.shape[:2] != frame.shape[:2]:
                    plate = cv2.resize(plate, (frame.shape[1], frame.shape[0]))
                matte = background_tool.soft_matte(
                    frame, mask_service.load_mask_alpha(project_dir / "masks", req.frame_index)
                )[..., None]
                output = matte * frame.astype(np.float32) + (1 - matte) * plate.astype(np.float32)
                ok, encoded = cv2.imencode(".jpg", np.clip(output, 0, 255).astype(np.uint8),
                                           [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise RuntimeError("Background preview could not be encoded")
                jpeg = encoded.tobytes()
            pending_dir.mkdir(parents=True, exist_ok=True)
            pending_path.write_bytes(generated)
            project_manager.update_status(
                req.project_id,
                pending_edit_preview={
                    "edit_type": req.edit_type,
                    "frame_index": req.frame_index,
                    "prompt": req.prompt or "",
                },
            )
        else:
            pending_path.unlink(missing_ok=True)
            project_manager.update_status(req.project_id, pending_edit_preview=None)
            jpeg = await asyncio.to_thread(
                preview_service.render_preview,
                req.project_id, req.frame_index, req.edit_type,
                color=req.color, blur_strength=req.blur_strength,
                scale=req.scale, dx=req.dx or 0, dy=req.dy or 0)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": str(e)}, status_code=400)
    return Response(content=jpeg, media_type="image/jpeg")


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


@app.post("/edit/preview/cancel")
async def cancel_edit_preview(req: CancelRequest):
    """Discard a prepared keyframe preview without touching project frames."""
    project_dir = project_manager.get_project_dir(req.project_id)
    (project_dir / "pending" / "edit_anchor.jpg").unlink(missing_ok=True)
    project_manager.update_status(req.project_id, pending_edit_preview=None)
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

    await asyncio.to_thread(
        supabase_storage_service.sync_export, req.project_id, output_path)

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
