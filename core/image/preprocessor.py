"""
Image preprocessing: loading, normalization, format handling.
"""
import numpy as np
import base64
import io
import time
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass, field
from PIL import Image, ExifTags

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import IMAGE_MAX_DIMENSION


@dataclass
class ImageData:
    """Container for preprocessed image data used by all analyzers."""
    pil_image: Image.Image          # RGB PIL Image (resized if needed)
    np_array: np.ndarray            # HxWxC uint8 numpy array
    grayscale: np.ndarray           # HxW float64 grayscale (0-1)
    raw_bytes: bytes                # Original file bytes (for compression analysis)
    original_format: str            # "JPEG", "PNG", "WEBP", etc.
    original_size: Tuple[int, int]  # (width, height) before resize
    final_size: Tuple[int, int]     # (width, height) after resize
    was_resized: bool
    metadata: dict = field(default_factory=dict)  # EXIF and other metadata


def load_image_from_path(file_path: str) -> ImageData:
    """Load an image from a file path."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")

    raw_bytes = path.read_bytes()
    pil_image = Image.open(io.BytesIO(raw_bytes))
    return _process_pil_image(pil_image, raw_bytes)


def load_image_from_base64(b64_string: str, image_format: str = "jpeg") -> ImageData:
    """Load an image from a base64-encoded string."""
    raw_bytes = base64.b64decode(b64_string)
    pil_image = Image.open(io.BytesIO(raw_bytes))
    return _process_pil_image(pil_image, raw_bytes)


def _process_pil_image(pil_image: Image.Image, raw_bytes: bytes) -> ImageData:
    """Process a PIL image into the standardized ImageData container."""
    original_format = pil_image.format or "UNKNOWN"
    original_size = pil_image.size  # (width, height)

    # Extract metadata before any processing
    metadata = _extract_metadata(pil_image)

    # Convert to RGB (handle RGBA, grayscale, CMYK, palette)
    if pil_image.mode == "RGBA":
        # Composite onto white background
        background = Image.new("RGB", pil_image.size, (255, 255, 255))
        background.paste(pil_image, mask=pil_image.split()[3])
        pil_image = background
    elif pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Resize if too large (preserve aspect ratio)
    was_resized = False
    w, h = pil_image.size
    if max(w, h) > IMAGE_MAX_DIMENSION:
        ratio = IMAGE_MAX_DIMENSION / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        pil_image = pil_image.resize(new_size, Image.LANCZOS)
        was_resized = True

    final_size = pil_image.size

    # Create numpy arrays
    np_array = np.array(pil_image, dtype=np.uint8)

    # Grayscale (float64, 0-1 range) using luminosity formula
    grayscale = (
        0.2989 * np_array[:, :, 0].astype(np.float64)
        + 0.5870 * np_array[:, :, 1].astype(np.float64)
        + 0.1140 * np_array[:, :, 2].astype(np.float64)
    ) / 255.0

    return ImageData(
        pil_image=pil_image,
        np_array=np_array,
        grayscale=grayscale,
        raw_bytes=raw_bytes,
        original_format=original_format,
        original_size=original_size,
        final_size=final_size,
        was_resized=was_resized,
        metadata=metadata,
    )


def _extract_metadata(pil_image: Image.Image) -> dict:
    """Extract EXIF and other metadata from the image."""
    metadata = {
        "format": pil_image.format,
        "mode": pil_image.mode,
        "size": pil_image.size,
        "exif_present": False,
        "exif_tags": 0,
        "exif_data": {},
    }

    try:
        exif_data = pil_image._getexif()
        if exif_data:
            metadata["exif_present"] = True
            metadata["exif_tags"] = len(exif_data)
            # Decode tag names
            decoded = {}
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                # Skip binary data, keep text/numeric values
                if isinstance(value, (str, int, float, tuple)):
                    decoded[tag_name] = value
                elif isinstance(value, bytes) and len(value) < 100:
                    decoded[tag_name] = value.hex()
            metadata["exif_data"] = decoded

            # Extract key camera info
            metadata["camera_make"] = decoded.get("Make", None)
            metadata["camera_model"] = decoded.get("Model", None)
            metadata["software"] = decoded.get("Software", None)
            metadata["datetime"] = decoded.get("DateTime", None)
    except Exception:
        pass

    # Check for ICC color profile
    try:
        icc = pil_image.info.get("icc_profile")
        metadata["has_color_profile"] = icc is not None
    except Exception:
        metadata["has_color_profile"] = False

    return metadata
