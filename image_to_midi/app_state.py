# app_state.py
"""Global state for the image-to-MIDI application."""

import cv2


# Global state
_IMAGES = {}


def register_image(image, image_id="default"):
    """Register an image for caching."""
    _IMAGES[image_id] = image
    return image_id


def get_image_by_id(image_id):
    """Get a registered image by ID."""
    return _IMAGES.get(image_id)


def get_image_id(image):
    """Get or create ID for an image."""
    # Simple hash-based ID
    image_hash = hash(image.tobytes())
    return register_image(image, str(image_hash))


# Function to load the fixed image
def load_fixed_image(path="img/paint_splatter.png"):
    """Load the fixed image for the app."""
    image = cv2.imread(path)
    if image is not None:
        # Convert from BGR to RGB for Gradio
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return register_image(image)
    return None
