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


def _safe(text: str) -> str:
    """Sanitize user text for Cloudinary effect parameters."""
    return text.replace(" ", "_").replace(";", "").replace("/", "").replace(":", "")


# ── Core edit types (use SAM 2 mask) ──────────────────────────────────

async def apply_replace(frame_public_id: str, mask_public_id: str, prompt: str) -> str:
    """Replace object with AI-generated content. Masking done locally."""
    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[
            {"effect": f"gen_replace:from_auto;to_{_safe(prompt)}"},
        ],
        secure=True,
    )
    return url


async def apply_resize(frame_public_id: str, mask_public_id: str,
                        x: int, y: int, w: int, h: int, scale: float) -> str:
    """Crop the object region and scale it."""
    new_w = int(w * scale)
    new_h = int(h * scale)
    offset_x = x - (new_w - w) // 2
    offset_y = y - (new_h - h) // 2

    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[
            {"overlay": frame_public_id.replace("/", ":"),
             "crop": "crop", "x": x, "y": y, "width": w, "height": h},
            {"width": new_w, "height": new_h, "crop": "scale"},
            {"flags": "layer_apply", "x": offset_x, "y": offset_y, "gravity": "north_west"},
        ],
        secure=True,
    )
    return url


async def apply_delete(frame_public_id: str, mask_public_id: str) -> str:
    """Remove object using gen_remove. Masking done locally after download."""
    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[
            {"effect": "gen_remove"},
        ],
        secure=True,
    )
    return url


async def apply_add(frame_public_id: str, prompt: str, x: int, y: int, w: int, h: int) -> str:
    """Generate and add a new object from a text prompt at the specified region."""
    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[
            {"effect": f"gen_fill:prompt_{_safe(prompt)}",
             "crop": "fill", "gravity": "north_west",
             "x": x, "y": y, "width": w, "height": h},
        ],
        secure=True,
    )
    return url


# ── Additional Cloudinary AI features ─────────────────────────────────

async def apply_background_remove(frame_public_id: str) -> str:
    """Remove the background, leaving only foreground objects."""
    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[{"effect": "background_removal"}],
        secure=True,
    )
    return url


async def apply_background_replace(frame_public_id: str, prompt: str) -> str:
    """Replace the background with AI-generated scene from a text prompt."""
    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[
            {"effect": f"gen_background_replace:prompt_{_safe(prompt)}"},
        ],
        secure=True,
    )
    return url


async def apply_generative_fill(frame_public_id: str, prompt: str = None) -> str:
    """Extend or fill image regions using generative AI."""
    effect = f"gen_fill:prompt_{_safe(prompt)}" if prompt else "gen_fill"
    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[{"effect": effect}],
        secure=True,
    )
    return url


async def apply_enhance(frame_public_id: str) -> str:
    """AI image enhancement — improve quality and details."""
    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[{"effect": "enhance"}],
        secure=True,
    )
    return url


async def apply_upscale(frame_public_id: str) -> str:
    """AI upscale — increase resolution while preserving quality."""
    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[{"effect": "upscale"}],
        secure=True,
    )
    return url


async def apply_restore(frame_public_id: str) -> str:
    """Generative restore — fix artifacts, noise, and compression damage."""
    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[{"effect": "gen_restore"}],
        secure=True,
    )
    return url


async def apply_blur(frame_public_id: str, strength: int = 500) -> str:
    """Apply blur effect to the frame."""
    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[{"effect": f"blur:{strength}"}],
        secure=True,
    )
    return url


async def apply_blur_region(frame_public_id: str, mask_public_id: str) -> str:
    """Blur only the masked region (e.g. faces, license plates)."""
    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[
            {"overlay": mask_public_id.replace("/", ":"), "effect": "blur:1000"},
            {"flags": "layer_apply"},
        ],
        secure=True,
    )
    return url


async def apply_drop_shadow(frame_public_id: str) -> str:
    """Add a drop shadow to the main subject."""
    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[
            {"effect": "background_removal"},
            {"effect": "dropshadow:50", "x": 10, "y": 10},
        ],
        secure=True,
    )
    return url


async def apply_generative_recolor(frame_public_id: str, prompt: str, color: str) -> str:
    """Generative recolor — recolor a specific object identified by prompt."""
    url = cloudinary.CloudinaryImage(frame_public_id).build_url(
        transformation=[
            {"effect": f"gen_recolor:prompt_{_safe(prompt)};to-color_{color}"},
        ],
        secure=True,
    )
    return url


async def download_url(url: str, save_path: Path):
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        save_path.write_bytes(resp.content)
