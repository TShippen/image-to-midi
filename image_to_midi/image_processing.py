"""Image preprocessing functions for the image-to-MIDI pipeline.

This module provides functions for converting input images into binary masks
suitable for musical note detection. The primary operation is thresholding
to isolate dark regions (paint splatters or musical notation) from the
background.
"""

import cv2
import numpy as np


def mask_image(image: np.ndarray, threshold_value: int) -> np.ndarray:
    """Convert a BGR image to a binary mask using inverted thresholding.

    Converts a color image to grayscale and applies inverse binary thresholding
    to create a mask where dark regions (paint splatters, musical notation)
    become white pixels and light regions become black pixels. This prepares
    the image for contour-based note detection.

    Args:
        image: Input BGR image as a 3-channel NumPy array.
        threshold_value: Grayscale threshold value (0-255) for binarization.
                        Pixels darker than this become white in the output.

    Returns:
        Binary image as a 2D uint8 NumPy array where detected features
        are white (255) and background is black (0).
    """
    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply inverse binary threshold to make dark regions white
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    return binary
