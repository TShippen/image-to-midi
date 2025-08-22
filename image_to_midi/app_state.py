"""Application state management for image registration and caching.

This module provides functionality for registering and retrieving images
by unique identifiers, enabling efficient image caching and state management
across the application. Images are identified by CRC32 checksums of their
binary data.
"""

import zlib
import cv2
import numpy as np

# In-memory registry of images by ID
_image_registry: dict[str, np.ndarray] = {}


def load_fixed_image(path: str = "img/paint_splatter.png") -> np.ndarray | None:
    """Load a fixed image file as an RGB NumPy array.

    Loads an image from the specified path and converts it from BGR
    (OpenCV's default) to RGB format for use with Gradio and other
    display systems.

    Args:
        path: File path to the image (default "img/paint_splatter.png").

    Returns:
        RGB image as a NumPy array, or None if the file cannot be read.
    """
    image = cv2.imread(path)
    if image is not None:
        # Convert BGR→RGB for Gradio
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return None


def register_image(image: np.ndarray, image_id: str | None = None) -> str:
    """Register an image in the global registry with a unique identifier.

    Stores the image in an in-memory registry for later retrieval. If no
    ID is provided, generates a unique identifier using CRC32 hash of the
    image's binary data.

    Args:
        image: NumPy array representing the image data.
        image_id: Optional unique identifier for the image. If None, a
                 CRC32-based ID will be generated.

    Returns:
        The image identifier (either provided or generated) as a string.
    """
    if image_id is None:
        # Generate unique ID from image data using CRC32
        data = image.tobytes()
        crc = zlib.crc32(data) & 0xFFFFFFFF
        image_id = f"img_{crc:08x}"

    _image_registry[image_id] = image
    return image_id


def get_image_id(image: np.ndarray) -> str:
    """Register an image and return its identifier.

    Convenience function that registers an image in the global registry
    and returns its generated or existing identifier.

    Args:
        image: NumPy array representing the image data.

    Returns:
        The image identifier as a string.
    """
    return register_image(image)


def get_image_by_id(image_id: str) -> np.ndarray | None:
    """Retrieve a registered image by its identifier.

    Args:
        image_id: Unique identifier for the image.

    Returns:
        The registered image as a NumPy array, or None if not found.
    """
    return _image_registry.get(image_id)
