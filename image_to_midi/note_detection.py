import cv2
import numpy as np

def detect_notes(
    binary: np.ndarray,
    min_area: float,
    max_area: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float
) -> list[tuple[int,int,int,int]]:
    """Detect paint-blob notes via contour filtering.

    Args:
        binary: 2D binary image (uint8).
        min_area: Minimum contour area.
        max_area: Maximum contour area.
        min_aspect_ratio: Minimum width/height.
        max_aspect_ratio: Maximum width/height.

    Returns:
        List of bounding boxes (x, y, w, h) for valid blobs.
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    notes: list[tuple[int,int,int,int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect = w / h
        if min_area < area < max_area and min_aspect_ratio < aspect < max_aspect_ratio:
            notes.append((x, y, w, h))
    return notes
