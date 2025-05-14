# image_to_midi/app_state.py

import zlib
import cv2
from typing import Dict, Any

# In-memory registry of images by ID
_image_registry: Dict[str, Any] = {}


def load_fixed_image(path: str = "img/paint_splatter.png"):
    """
    Load the fixed image as an RGB NumPy array.
    Returns None if the file can't be read.
    """
    image = cv2.imread(path)
    if image is not None:
        # Convert BGR→RGB for Gradio
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return None


def register_image(image: Any, image_id: str = None) -> str:
    """
    Register `image` under `image_id`.
    If no ID is given, generate one via CRC32 of its bytes.
    """
    if image_id is None:
        data = image.tobytes()
        crc = zlib.crc32(data) & 0xFFFFFFFF
        image_id = f"img_{crc:08x}"
    _image_registry[image_id] = image
    return image_id


def get_image_id(image: Any) -> str:
    """Register `image` (if needed) and return its ID."""
    return register_image(image)


def get_image_by_id(image_id: str) -> Any:
    """Fetch the NumPy array for a given ID (or None)."""
    return _image_registry.get(image_id)
