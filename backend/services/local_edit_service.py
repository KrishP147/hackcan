"""Local image editing service - no Cloudinary dependency."""
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
import cv2
from typing import Optional

def _load_mask(mask_path: Path, target_shape: tuple) -> np.ndarray:
    """Load and resize mask to target shape."""
    mask = np.array(Image.open(mask_path).convert("L"))
    if mask.shape[:2] != target_shape[:2]:
        mask_img = Image.fromarray(mask).resize((target_shape[1], target_shape[0]), Image.NEAREST)
        mask = np.array(mask_img)
    return mask

def _get_mask_alpha(mask_array: np.ndarray) -> np.ndarray:
    """Convert mask to alpha channel (0-1 float)."""
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]
    return (mask_array > 128).astype(np.float32)[:, :, np.newaxis]

def apply_recolor(frame_path: Path, mask_path: Path, color_hex: str) -> None:
    """Recolor object using mask."""
    original = np.array(Image.open(frame_path).convert("RGB"))
    mask = _load_mask(mask_path, original.shape)
    alpha = _get_mask_alpha(mask)
    
    # Parse hex color
    color_hex = color_hex.lstrip("#")
    r, g, b = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)
    tint = np.full_like(original, [r, g, b])
    
    # Blend: 60% original + 40% tint in masked region
    tinted = (0.6 * original + 0.4 * tint).clip(0, 255).astype(np.uint8)
    result = (alpha * tinted + (1 - alpha) * original).astype(np.uint8)
    
    Image.fromarray(result).save(str(frame_path), quality=95)

def apply_blur_region(frame_path: Path, mask_path: Path, strength: int = 10) -> None:
    """Blur only the masked region."""
    original = np.array(Image.open(frame_path).convert("RGB"))
    # Feather the boundary so the blur does not produce a hard rectangular
    # cut line as the tracked mask moves frame-to-frame.
    alpha = _feathered_alpha(mask_path, original.shape, feather_px=3)[..., None]
    
    # Apply blur to entire image
    img = Image.fromarray(original)
    blurred = np.array(img.filter(ImageFilter.GaussianBlur(radius=strength)))
    
    # Composite: blurred where mask, original elsewhere
    result = (alpha * blurred + (1 - alpha) * original).astype(np.uint8)
    Image.fromarray(result).save(str(frame_path), quality=95)

def apply_resize(frame_path: Path, mask_path: Path, scale: float) -> None:
    """Resize object within mask."""
    original = np.array(Image.open(frame_path).convert("RGB"))
    mask = _load_mask(mask_path, original.shape)
    alpha = _get_mask_alpha(mask)
    
    # Find bounding box of mask
    rows = np.any(alpha[:, :, 0] > 0.5, axis=1)
    cols = np.any(alpha[:, :, 0] > 0.5, axis=0)
    if not rows.any() or not cols.any():
        return
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Extract object region and mask
    obj_region = original[y_min:y_max+1, x_min:x_max+1].copy()
    obj_mask_region = mask[y_min:y_max+1, x_min:x_max+1]
    
    # Calculate new dimensions
    old_h, old_w = obj_region.shape[:2]
    new_h = max(1, int(old_h * scale))
    new_w = max(1, int(old_w * scale))
    
    # Resize object
    obj_img = Image.fromarray(obj_region)
    resized_obj = np.array(obj_img.resize((new_w, new_h), Image.LANCZOS))
    
    # Resize mask
    mask_img = Image.fromarray(obj_mask_region)
    resized_mask = np.array(mask_img.resize((new_w, new_h), Image.NEAREST))
    resized_alpha = (resized_mask > 128).astype(np.float32)
    if resized_alpha.ndim == 2:
        resized_alpha = resized_alpha[:, :, np.newaxis]
    
    # Calculate center of original bounding box
    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2
    
    # Calculate new bounding box centered on the same point
    new_y_min = max(0, center_y - new_h // 2)
    new_y_max = min(original.shape[0], new_y_min + new_h)
    new_x_min = max(0, center_x - new_w // 2)
    new_x_max = min(original.shape[1], new_x_min + new_w)
    
    # Adjust if we hit boundaries
    if new_y_max - new_y_min < new_h:
        new_y_min = max(0, new_y_max - new_h)
    if new_x_max - new_x_min < new_w:
        new_x_min = max(0, new_x_max - new_w)
    
    # Crop resized object and mask to fit within image bounds
    crop_y_start = max(0, -new_y_min)
    crop_y_end = new_h - max(0, new_y_max - original.shape[0])
    crop_x_start = max(0, -new_x_min)
    crop_x_end = new_w - max(0, new_x_max - original.shape[1])
    
    resized_obj_crop = resized_obj[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    resized_alpha_crop = resized_alpha[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    
    # Final placement coordinates
    final_y_min = max(0, new_y_min)
    final_y_max = min(original.shape[0], new_y_min + resized_obj_crop.shape[0])
    final_x_min = max(0, new_x_min)
    final_x_max = min(original.shape[1], new_x_min + resized_obj_crop.shape[1])
    
    # Ensure dimensions match
    final_h = final_y_max - final_y_min
    final_w = final_x_max - final_x_min
    
    if resized_obj_crop.shape[0] != final_h or resized_obj_crop.shape[1] != final_w:
        resized_obj_crop = resized_obj_crop[:final_h, :final_w]
        resized_alpha_crop = resized_alpha_crop[:final_h, :final_w]
    
    # Create result image
    result = original.copy()
    
    # Composite: resized object where mask is white, original elsewhere
    result[final_y_min:final_y_max, final_x_min:final_x_max] = (
        resized_alpha_crop * resized_obj_crop + 
        (1 - resized_alpha_crop) * result[final_y_min:final_y_max, final_x_min:final_x_max]
    ).astype(np.uint8)
    
    Image.fromarray(result).save(str(frame_path), quality=95)

def apply_remove(frame_path: Path, mask_path: Path) -> None:
    """Remove object by inpainting."""
    original = np.array(Image.open(frame_path).convert("RGB"))
    mask = _load_mask(mask_path, original.shape)
    
    # Convert to OpenCV format
    img_cv = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    mask_cv = (mask > 128).astype(np.uint8) * 255
    
    # Use inpainting to fill the masked region
    inpainted = cv2.inpaint(img_cv, mask_cv, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    result = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    Image.fromarray(result).save(str(frame_path), quality=95)



def apply_blur(frame_path: Path, strength: int = 10) -> None:
    """Apply blur to entire frame."""
    img = Image.open(frame_path).convert("RGB")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=strength))
    blurred.save(str(frame_path), quality=95)


def _feathered_alpha(mask_path: Path, target_shape: tuple, feather_px: int = 5) -> np.ndarray:
    """Mask as float alpha in [0,1] with a Gaussian-feathered boundary."""
    mask = _load_mask(mask_path, target_shape)
    a = (mask > 128).astype(np.float32)
    k = feather_px * 2 + 1
    return cv2.GaussianBlur(a, (k, k), 0)

def apply_color_pop(frame_path: Path, mask_path: Path) -> None:
    """Object stays color, everything else goes grayscale (§5.1)."""
    img = cv2.imread(str(frame_path))
    a = _feathered_alpha(mask_path, img.shape)[..., None]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    out = a * img.astype(np.float32) + (1 - a) * gray3
    cv2.imwrite(str(frame_path), np.clip(out, 0, 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 95])

def apply_glow(frame_path: Path, mask_path: Path,
               intensity: float = 0.6, radius: int = 21) -> None:
    """Bloom halo (§5.1): additive glow in the object's brightened color,
    carried by a blurred mask alpha so the falloff extends outside the mask."""
    img = cv2.imread(str(frame_path)).astype(np.float32)
    hard = (_load_mask(mask_path, img.shape) > 128).astype(np.float32)
    if hard.sum() > 0:
        mean_color = img[hard > 0].mean(axis=0)
    else:
        mean_color = np.array([255.0, 255.0, 255.0])
    glow_color = 0.5 * mean_color + 0.5 * 255.0           # push toward white
    halo_alpha = cv2.GaussianBlur(hard, (radius, radius), 0)
    out = img + intensity * halo_alpha[..., None] * glow_color[None, None, :]
    cv2.imwrite(str(frame_path), np.clip(out, 0, 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
