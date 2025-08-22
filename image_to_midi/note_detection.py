"""Note detection functions using contour analysis.

This module provides functions for detecting musical note-like shapes
in binary images using OpenCV contour detection and filtering. The
primary function identifies connected components that match specified
size and aspect ratio criteria, returning bounding boxes for valid notes.
"""

import cv2
import numpy as np


def detect_notes(
    binary: np.ndarray,
    min_area: float,
    max_area: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> list[tuple[int, int, int, int]]:
    """Detect musical note candidates using contour analysis and filtering.

    Finds connected components in a binary image and filters them based on
    area and aspect ratio criteria to identify shapes that could represent
    musical notes or paint splatters. Uses OpenCV's contour detection to
    find external boundaries only.

    Args:
        binary: Input binary image as 2D uint8 NumPy array where features
               are white (255) and background is black (0).
        min_area: Minimum allowable contour area in pixels.
        max_area: Maximum allowable contour area in pixels.
        min_aspect_ratio: Minimum width/height ratio for valid shapes.
        max_aspect_ratio: Maximum width/height ratio for valid shapes.

    Returns:
        List of bounding box tuples (x, y, w, h) representing detected notes,
        where each tuple contains integer pixel coordinates and dimensions.
    """
    # Find external contours only (no nested contours)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    notes: list[tuple[int, int, int, int]] = []

    # Filter contours based on area and aspect ratio criteria
    for contour in contours:
        # Get bounding rectangle for this contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate area and aspect ratio
        area = cv2.contourArea(contour)
        aspect_ratio = w / h

        # Accept contour if it meets all criteria
        if (
            min_area < area < max_area
            and min_aspect_ratio < aspect_ratio < max_aspect_ratio
        ):
            notes.append((x, y, w, h))

    return notes
