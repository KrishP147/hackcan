"""Durable project media storage.

Modal's Volume is a hot working cache. Supabase Storage owns the original,
current edited video, thumbnail, checkpoint, and exported MP4. All functions
are no-ops when Supabase credentials are not configured, preserving local-only
development.
"""
from __future__ import annotations

import json
import mimetypes
import os
import tarfile
import tempfile
from pathlib import Path
from urllib.parse import quote

import httpx

from services import ffmpeg_service, project_manager

BUCKET = os.getenv("SUPABASE_MEDIA_BUCKET", "project-media")


def _url() -> str:
    return os.getenv("SUPABASE_URL", "").rstrip("/")


def _key() -> str:
    return os.getenv("SUPABASE_SECRET_KEY", "")


def configured() -> bool:
    return bool(_url() and _key())


def object_path(project_id: str, name: str) -> str:
    return f"projects/{project_id}/{name}"


def _headers(content_type: str | None = None) -> dict[str, str]:
    headers = {
        "apikey": _key(),
        "Authorization": f"Bearer {_key()}",
    }
    if content_type:
        headers["Content-Type"] = content_type
        headers["x-upsert"] = "true"
    return headers


def upload_file(local_path: Path, remote_path: str, content_type: str | None = None) -> str | None:
    if not configured() or not local_path.exists():
        return None
    media_type = content_type or mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
    encoded_path = quote(remote_path, safe="/")
    endpoint = f"{_url()}/storage/v1/object/{BUCKET}/{encoded_path}"
    with local_path.open("rb") as source:
        response = httpx.post(
            endpoint,
            headers=_headers(media_type),
            content=source,
            timeout=httpx.Timeout(180.0),
        )
    response.raise_for_status()
    return remote_path


def patch_project(project_id: str, **updates) -> None:
    if not configured() or not updates:
        return
    endpoint = f"{_url()}/rest/v1/projects"
    response = httpx.patch(
        endpoint,
        params={"project_id": f"eq.{project_id}"},
        headers={**_headers(), "Content-Type": "application/json", "Prefer": "return=minimal"},
        json=updates,
        timeout=30.0,
    )
    if response.is_success:
        return
    # The media migration is backwards-compatible. Until its extra columns
    # exist, at least keep the legacy status current; object paths remain
    # deterministic and recoverable from the project id.
    legacy_updates = {key: value for key, value in updates.items() if key == "status"}
    if response.status_code == 400 and legacy_updates:
        fallback = httpx.patch(
            endpoint,
            params={"project_id": f"eq.{project_id}"},
            headers={**_headers(), "Content-Type": "application/json", "Prefer": "return=minimal"},
            json=legacy_updates,
            timeout=30.0,
        )
        fallback.raise_for_status()
        return
    response.raise_for_status()


def _write_checkpoint(project_id: str, destination: Path) -> None:
    project_dir = project_manager.get_project_dir(project_id)
    with tarfile.open(destination, "w:gz") as archive:
        status_path = project_dir / "status.json"
        if status_path.exists():
            archive.add(status_path, arcname="status.json")
        masks_dir = project_dir / "masks"
        if masks_dir.exists():
            for mask in sorted(masks_dir.glob("mask_*.png")):
                archive.add(mask, arcname=f"masks/{mask.name}")


def sync_checkpoint(project_id: str) -> str | None:
    if not configured():
        return None
    project_manager.update_status(project_id, storage_sync_status="syncing")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint.tar.gz"
            _write_checkpoint(project_id, checkpoint)
            remote = object_path(project_id, "checkpoint.tar.gz")
            upload_file(checkpoint, remote, "application/gzip")
        patch_project(project_id, checkpoint_path=remote)
        project_manager.update_status(
            project_id,
            storage_sync_status="done",
            storage_checkpoint_path=remote,
        )
        return remote
    except Exception as exc:
        project_manager.update_status(
            project_id, storage_sync_status="error", storage_sync_error=str(exc))
        print(f"[storage] checkpoint sync failed for {project_id}: {exc}")
        return None


def sync_extract_metadata(project_id: str) -> None:
    if not configured():
        return
    project_dir = project_manager.get_project_dir(project_id)
    status = project_manager.get_status(project_id)
    thumbnail = project_dir / "frames" / "frame_0001.jpg"
    remote = object_path(project_id, "thumbnail.jpg")
    try:
        upload_file(thumbnail, remote, "image/jpeg")
        patch_project(
            project_id,
            status="ready",
            thumbnail_path=remote,
            frame_count=int(status.get("frame_count") or 0),
        )
        project_manager.update_status(project_id, storage_thumbnail_path=remote)
        sync_checkpoint(project_id)
    except Exception as exc:
        project_manager.update_status(
            project_id, storage_sync_status="error", storage_sync_error=str(exc))
        print(f"[storage] extraction metadata sync failed for {project_id}: {exc}")


def sync_current_video(project_id: str) -> str | None:
    if not configured():
        return None
    project_dir = project_manager.get_project_dir(project_id)
    status = project_manager.get_status(project_id)
    current_video = project_dir / "current.mp4"
    project_manager.update_status(project_id, storage_sync_status="syncing")
    try:
        ffmpeg_service.encode_video(
            project_dir / "frames", current_video, fps=float(status.get("fps") or 30))
        remote = object_path(project_id, "current.mp4")
        upload_file(current_video, remote, "video/mp4")
        checkpoint = sync_checkpoint(project_id)
        patch_project(
            project_id,
            status="ready",
            current_path=remote,
            checkpoint_path=checkpoint,
            edit_version=int(status.get("edit_version") or 0),
        )
        project_manager.update_status(
            project_id,
            storage_sync_status="done",
            storage_current_path=remote,
        )
        return remote
    except Exception as exc:
        project_manager.update_status(
            project_id, storage_sync_status="error", storage_sync_error=str(exc))
        print(f"[storage] current video sync failed for {project_id}: {exc}")
        return None


def sync_export(project_id: str, output_path: Path) -> str | None:
    if not configured():
        return None
    try:
        remote = object_path(project_id, "exports/final.mp4")
        upload_file(output_path, remote, "video/mp4")
        patch_project(project_id, export_path=remote, status="ready")
        project_manager.update_status(project_id, storage_export_path=remote)
        return remote
    except Exception as exc:
        project_manager.update_status(
            project_id, storage_sync_status="error", storage_sync_error=str(exc))
        print(f"[storage] export sync failed for {project_id}: {exc}")
        return None


def backfill_volume() -> dict[str, object]:
    """Copy every existing Modal Volume project into durable storage."""
    base_dir = project_manager.BASE_DIR
    migrated: list[str] = []
    failed: dict[str, str] = {}
    if not configured() or not base_dir.exists():
        return {"migrated": migrated, "failed": failed, "configured": configured()}

    for project_dir in sorted(base_dir.iterdir()):
        if not project_dir.is_dir() or not re_full_project_id(project_dir.name):
            continue
        project_id = project_dir.name
        try:
            original = project_dir / "original.mp4"
            if original.exists():
                remote_original = object_path(project_id, "original.mp4")
                upload_file(original, remote_original, "video/mp4")
                current_status = project_manager.get_status(project_id)
                patch_project(
                    project_id,
                    original_path=remote_original,
                    storage_status="stored",
                    status=current_status.get("status") or "created",
                )

            if any((project_dir / "frames").glob("frame_*.jpg")):
                sync_extract_metadata(project_id)
                status = project_manager.get_status(project_id)
                if int(status.get("edit_version") or 0) > 0:
                    sync_current_video(project_id)
            migrated.append(project_id)
        except Exception as exc:
            failed[project_id] = str(exc)
            print(f"[storage] backfill failed for {project_id}: {exc}")

    return {"migrated": migrated, "failed": failed, "configured": True}


def re_full_project_id(value: str) -> bool:
    return len(value) == 32 and all(char in "0123456789abcdef" for char in value)
