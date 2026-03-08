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
    mask = _load_mask(mask_path, original.shape)
    alpha = _get_mask_alpha(mask)
    
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
    
    # Extract object region
    obj_region = original[y_min:y_max+1, x_min:x_max+1]
    obj_mask = alpha[y_min:y_max+1, x_min:x_min+1]
    
    # Resize object
    new_h = int(obj_region.shape[0] * scale)
    new_w = int(obj_region.shape[1] * scale)
    resized_obj = np.array(Image.fromarray(obj_region).resize((new_w, new_h), Image.LANCZOS))
    
    # Calculate new position (centered)
    offset_y = (new_h - obj_region.shape[0]) // 2
    offset_x = (new_w - obj_region.shape[1]) // 2
    
    # Create new mask for resized object
    resized_mask = np.array(Image.fromarray((mask[y_min:y_max+1, x_min:x_max+1] > 128).astype(np.uint8) * 255).resize((new_w, new_h), Image.NEAREST))
    resized_alpha = (resized_mask > 128).astype(np.float32)[:, :, np.newaxis]
    
    # Create result image
    result = original.copy()
    
    # Place resized object
    new_y_min = max(0, y_min - offset_y)
    new_y_max = min(original.shape[0], new_y_min + new_h)
    new_x_min = max(0, x_min - offset_x)
    new_x_max = min(original.shape[1], new_x_min + new_w)
    
    obj_y_start = max(0, offset_y - y_min)
    obj_y_end = obj_y_start + (new_y_max - new_y_min)
    obj_x_start = max(0, offset_x - x_min)
    obj_x_end = obj_x_start + (new_x_max - new_x_min)
    
    resized_crop = resized_obj[obj_y_start:obj_y_end, obj_x_start:obj_x_end]
    mask_crop = resized_alpha[obj_y_start:obj_y_end, obj_x_start:obj_x_end]
    
    result[new_y_min:new_y_max, new_x_min:new_x_max] = (
        mask_crop * resized_crop + (1 - mask_crop) * result[new_y_min:new_y_max, new_x_min:new_x_max]
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

def apply_enhance(frame_path: Path) -> None:
    """Enhance image quality."""
    img = Image.open(frame_path).convert("RGB")
    
    # Apply enhancements
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.2)
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.05)
    
    img.save(str(frame_path), quality=95)

def apply_upscale(frame_path: Path, scale: int = 2) -> None:
    """Upscale image using LANCZOS resampling."""
    img = Image.open(frame_path).convert("RGB")
    width, height = img.size
    upscaled = img.resize((width * scale, height * scale), Image.LANCZOS)
    upscaled.save(str(frame_path), quality=95)

def apply_blur(frame_path: Path, strength: int = 10) -> None:
    """Apply blur to entire frame."""
    img = Image.open(frame_path).convert("RGB")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=strength))
    blurred.save(str(frame_path), quality=95)

def apply_restore(frame_path: Path) -> None:
    """Restore image by reducing noise and enhancing."""
    img = Image.open(frame_path).convert("RGB")
    
    # Convert to numpy for processing
    img_array = np.array(img)
    
    # Apply denoising
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)
    img_array = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    
    # Enhance
    img = Image.fromarray(img_array)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.1)
    
    img.save(str(frame_path), quality=95)
