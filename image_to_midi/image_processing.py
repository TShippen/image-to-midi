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
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return binary
