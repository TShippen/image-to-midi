import cv2
import numpy as np

def preprocess_image(image: np.ndarray, threshold_value: int) -> np.ndarray:
    """Convert a BGR image to a binary (inverted) mask.

    Args:
        image: 3-channel BGR image as a NumPy array.
        threshold_value: Threshold in [0,255] for binarization.

    Returns:
        2D uint8 binary image where splatters are white.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, threshold_value, 255, cv2.THRESH_BINARY_INV
    )
    return binary

def create_note_visualization(
    original: np.ndarray,
    binary: np.ndarray,
    notes: list[tuple[int,int,int,int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Overlay bounding boxes on both RGB and binary images.

    Args:
        original: BGR image array.
        binary: 2D binary image.
        notes: List of (x, y, w, h) rectangles.

    Returns:
        Tuple of (RGB with boxes, Binary→RGB with boxes).
    """
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    bin_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    for x, y, w, h in notes:
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(bin_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return rgb, bin_rgb
