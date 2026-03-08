import os
import base64
from pathlib import Path
from google import genai
from google.genai import types
from google.genai.types import FinishReason
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from io import BytesIO

# Load environment variables from backend/.env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

_client = None


def _get_client():
    global _client
    if _client is None:
        # Try GOOGLE_API_KEY first (takes precedence), then GEMINI_API_KEY
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print(f"ERROR: API key not found. Looking for .env at: {env_path}")
            print(f"Make sure either GOOGLE_API_KEY or GEMINI_API_KEY is set in {env_path}")
            print(f"Current env vars: {list(os.environ.keys())}")
            raise RuntimeError("API key not set in environment. Please add GOOGLE_API_KEY or GEMINI_API_KEY to your .env file.")
        
        print(f"Using API key (length: {len(api_key)}, starts with: {api_key[:4]}...)")
        try:
            _client = genai.Client(api_key=api_key)
            print("Gemini client initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Gemini client: {e}")
            raise RuntimeError(f"Failed to initialize Gemini client: {e}")
    return _client


async def edit_frame(frame_path: Path, prompt: str, mask_path: Path | None = None) -> bytes:
    """Send a frame image + text prompt to Gemini and return the edited image bytes."""
    import asyncio
    
    def _generate_sync():
        """Synchronous wrapper for the Gemini API call."""
        client = _get_client()

        # Read frame as bytes
        frame_bytes = frame_path.read_bytes()

        # Try the image generation model first
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_bytes(data=frame_bytes, mime_type="image/jpeg"),
                            types.Part.from_text(
                                text=f"Edit this image: {prompt}. Return only the edited image, keep the same dimensions and aspect ratio."
                            ),
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )
        except Exception as e:
            # Fallback to regular flash model if image model doesn't work
            print(f"Image model failed, trying flash model: {str(e)}")
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_bytes(data=frame_bytes, mime_type="image/jpeg"),
                            types.Part.from_text(
                                text=f"Edit this image: {prompt}. Return only the edited image, keep the same dimensions and aspect ratio."
                            ),
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )
        return response
    
    try:
        # Run the synchronous API call in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _generate_sync)

        # Check for errors in response
        if not response.candidates or len(response.candidates) == 0:
            error_msg = "No candidates in response"
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                error_msg += f": {response.prompt_feedback}"
            raise RuntimeError(error_msg)

        # Check for finish reason errors
        candidate = response.candidates[0]
        if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
            finish_reason = candidate.finish_reason
            # STOP means success, other values indicate errors
            if finish_reason != FinishReason.STOP and finish_reason != 1:
                error_msg = f"Generation stopped with error: finish_reason={finish_reason}"
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    error_msg += f", safety_ratings={candidate.safety_ratings}"
                raise RuntimeError(error_msg)

        # Extract image from response
        if not hasattr(candidate, 'content') or not candidate.content.parts:
            raise RuntimeError("No content parts in response")

        edited_bytes = None
        for part in candidate.content.parts:
            if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.mime_type.startswith("image/"):
                edited_bytes = part.inline_data.data
                break
        
        if not edited_bytes:
            raise RuntimeError("Gemini did not return an image in the response")
        
        # If mask is provided, composite the edited result with the original using the mask
        if mask_path and mask_path.exists():
            mask_array = np.array(Image.open(mask_path).convert("L"))
            edited_bytes = _composite_ai_edit_with_mask(frame_path, edited_bytes, mask_array)
        
        return edited_bytes
    except Exception as e:
        # Re-raise with more context
        error_str = str(e)
        if isinstance(e, RuntimeError):
            raise
        # Check for common API errors
        if "API key not valid" in error_str or "INVALID_ARGUMENT" in error_str:
            raise RuntimeError(f"Invalid API key. Please verify your GEMINI_API_KEY in .env file. Error: {error_str}")
        elif "quota" in error_str.lower() or "QUOTA_EXCEEDED" in error_str:
            raise RuntimeError(f"Quota exceeded. Image generation requires a paid API key with billing enabled. Error: {error_str}")
        elif "permission" in error_str.lower() or "PERMISSION_DENIED" in error_str:
            raise RuntimeError(f"Permission denied. Make sure your API key has access to image generation models. Error: {error_str}")
        else:
            raise RuntimeError(f"Gemini API error: {error_str}")


def _composite_ai_edit_with_mask(original_path: Path, edited_bytes: bytes, mask_array: np.ndarray) -> bytes:
    """Composite AI-edited frame onto original using mask. Returns edited image bytes.
    
    Args:
        original_path: Path to original frame
        edited_bytes: Bytes of AI-edited image
        mask_array: Binary mask array (white pixels = object, black = background)
    
    Returns:
        Composited image as bytes
    """
    # Load original frame
    original = np.array(Image.open(original_path).convert("RGB"))
    
    # Decode edited bytes to numpy array
    from io import BytesIO
    edited_img = Image.open(BytesIO(edited_bytes)).convert("RGB")
    edited = np.array(edited_img)
    
    # Resize mask to match frame dimensions if needed
    mask = mask_array
    if mask.shape[:2] != original.shape[:2]:
        mask_img = Image.fromarray(mask).resize((original.shape[1], original.shape[0]), Image.NEAREST)
        mask = np.array(mask_img)
    
    # Normalize mask to 0-1 float, expand to 3 channels
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    alpha = (mask > 0).astype(np.float32)[:, :, np.newaxis]
    
    # Resize edited to match original if dimensions differ
    if edited.shape[:2] != original.shape[:2]:
        edited_img = Image.fromarray(edited).resize((original.shape[1], original.shape[0]), Image.LANCZOS)
        edited = np.array(edited_img)
    
    # Composite: edited where mask=white, original elsewhere
    result = (alpha * edited + (1 - alpha) * original).astype(np.uint8)
    
    # Convert back to bytes
    result_img = Image.fromarray(result)
    output = BytesIO()
    result_img.save(output, format="JPEG", quality=95)
    return output.getvalue()


async def edit_frame_with_reference(frame_path: Path, prompt: str, reference_frame_path: Path | None = None, mask_path: Path | None = None) -> bytes:
    """Edit a frame using Gemini, optionally with a reference frame for style consistency.
    
    Args:
        frame_path: Path to the target frame to transform
        prompt: Edit instruction text
        reference_frame_path: Optional path to a reference frame to use as style guide
        mask_path: Optional path to mask file - if provided, edits will only apply to masked region
    
    Returns:
        Edited image bytes
    """
    import asyncio
    
    def _generate_sync():
        """Synchronous wrapper for the Gemini API call."""
        client = _get_client()

        # Read target frame as bytes
        frame_bytes = frame_path.read_bytes()

        parts = [
            types.Part.from_bytes(data=frame_bytes, mime_type="image/jpeg"),
        ]

        # If reference frame is provided, include it with instruction to match style
        if reference_frame_path and reference_frame_path.exists():
            reference_bytes = reference_frame_path.read_bytes()
            parts.append(types.Part.from_bytes(data=reference_bytes, mime_type="image/jpeg"))
            parts.append(
                types.Part.from_text(
                    text=f"Apply the same transformation style from the reference image to this frame. "
                         f"Original edit instruction: {prompt}. "
                         f"Return only the edited image, keep the same dimensions and aspect ratio as the target frame."
                )
            )
        else:
            parts.append(
                types.Part.from_text(
                    text=f"Edit this image: {prompt}. Return only the edited image, keep the same dimensions and aspect ratio."
                )
            )

        # Try the image generation model first
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                contents=[
                    types.Content(parts=parts)
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )
        except Exception as e:
            # Fallback to regular flash model if image model doesn't work
            print(f"Image model failed, trying flash model: {str(e)}")
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[
                    types.Content(parts=parts)
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )
        return response
    
    try:
        # Run the synchronous API call in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _generate_sync)

        # Check for errors in response
        if not response.candidates or len(response.candidates) == 0:
            error_msg = "No candidates in response"
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                error_msg += f": {response.prompt_feedback}"
            raise RuntimeError(error_msg)

        # Check for finish reason errors
        candidate = response.candidates[0]
        if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
            finish_reason = candidate.finish_reason
            # STOP means success, other values indicate errors
            if finish_reason != FinishReason.STOP and finish_reason != 1:
                error_msg = f"Generation stopped with error: finish_reason={finish_reason}"
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    error_msg += f", safety_ratings={candidate.safety_ratings}"
                raise RuntimeError(error_msg)

        # Extract image from response
        if not hasattr(candidate, 'content') or not candidate.content.parts:
            raise RuntimeError("No content parts in response")

        edited_bytes = None
        for part in candidate.content.parts:
            if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.mime_type.startswith("image/"):
                edited_bytes = part.inline_data.data
                break
        
        if not edited_bytes:
            raise RuntimeError("Gemini did not return an image in the response")
        
        # If mask is provided, composite the edited result with the original using the mask
        if mask_path and mask_path.exists():
            mask_array = np.array(Image.open(mask_path).convert("L"))
            edited_bytes = _composite_ai_edit_with_mask(frame_path, edited_bytes, mask_array)
        
        return edited_bytes
    except Exception as e:
        # Re-raise with more context
        error_str = str(e)
        if isinstance(e, RuntimeError):
            raise
        # Check for common API errors
        if "API key not valid" in error_str or "INVALID_ARGUMENT" in error_str:
            raise RuntimeError(f"Invalid API key. Please verify your GEMINI_API_KEY in .env file. Error: {error_str}")
        elif "quota" in error_str.lower() or "QUOTA_EXCEEDED" in error_str:
            raise RuntimeError(f"Quota exceeded. Image generation requires a paid API key with billing enabled. Error: {error_str}")
        elif "permission" in error_str.lower() or "PERMISSION_DENIED" in error_str:
            raise RuntimeError(f"Permission denied. Make sure your API key has access to image generation models. Error: {error_str}")
        else:
            raise RuntimeError(f"Gemini API error: {error_str}")
