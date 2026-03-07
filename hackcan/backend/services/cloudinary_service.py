import os
import cloudinary
import cloudinary.uploader
import cloudinary.api

def configure():
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True,
    )

def upload_file(file_path: str, folder: str = "frameshift", resource_type: str = "image") -> dict:
    result = cloudinary.uploader.upload(
        file_path,
        folder=folder,
        resource_type=resource_type,
    )
    return {
        "public_id": result["public_id"],
        "url": result["secure_url"],
        "width": result.get("width"),
        "height": result.get("height"),
    }

def get_url(public_id: str, transformations: list = None, resource_type: str = "image") -> str:
    options = {"secure": True}
    if transformations:
        options["transformation"] = transformations
    return cloudinary.CloudinaryImage(public_id).build_url(**options)


import httpx
from pathlib import Path


async def apply_recolor(frame_public_id: str, mask_public_id: str, color: str) -> str:
    """Colorize the masked region. Opens mask as overlay layer, colorizes it, applies."""
    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[
            {"overlay": mask_public_id.replace("/", ":"), "effect": f"colorize:80", "color": f"#{color}"},
            {"flags": "layer_apply", "blend_mode": "multiply"},
        ],
        secure=True,
    )
    return url


async def apply_replace(frame_public_id: str, replacement_public_id: str,
                         x: int, y: int, w: int, h: int) -> str:
    """Overlay replacement image at bbox position."""
    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[
            {"overlay": replacement_public_id.replace("/", ":"),
             "width": w, "height": h, "crop": "scale"},
            {"flags": "layer_apply", "x": x, "y": y, "gravity": "north_west"},
        ],
        secure=True,
    )
    return url


async def apply_resize(frame_public_id: str, mask_public_id: str,
                        x: int, y: int, w: int, h: int, scale: float) -> str:
    """Scale the masked region by factor and overlay back."""
    new_w = int(w * scale)
    new_h = int(h * scale)
    offset_x = x - (new_w - w) // 2
    offset_y = y - (new_h - h) // 2

    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[
            {"overlay": mask_public_id.replace("/", ":"),
             "width": new_w, "height": new_h, "crop": "scale"},
            {"flags": "layer_apply", "x": offset_x, "y": offset_y, "gravity": "north_west"},
        ],
        secure=True,
    )
    return url


async def apply_delete(frame_public_id: str, mask_public_id: str) -> str:
    """Remove masked object using Cloudinary generative AI (gen_remove)."""
    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[
            {"effect": "gen_remove"},
        ],
        secure=True,
    )
    return url


async def apply_add(frame_public_id: str, prompt: str, x: int, y: int, w: int, h: int) -> str:
    """Generate object from prompt using gen_replace. Replaces background region with prompt object."""
    safe_prompt = prompt.replace(" ", "_").replace(";", "").replace("/", "")
    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[
            {"effect": f"gen_replace:from_background;to_{safe_prompt}"},
        ],
        secure=True,
    )
    return url


async def download_url(url: str, save_path: Path):
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        save_path.write_bytes(resp.content)
