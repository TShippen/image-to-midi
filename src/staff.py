import cv2
import numpy as np

def detect_lines(
    notes: list[tuple[int,int,int,int]],
    num_lines: int
) -> np.ndarray:
    """Compute equally-spaced staff lines between blobs’ centers.

    Args:
        notes: List of (x, y, w, h).
        num_lines: Desired number of horizontal lines.

    Returns:
        1D array of Y-positions.
    """
    centers = [y + h / 2 for _, y, _, h in notes]
    top, bottom = min(centers), max(centers)
    return np.linspace(top, bottom, num_lines)

def average_blob_height(
    notes: list[tuple[int,int,int,int]]
) -> list[tuple[int,float,int,float]]:
    """Normalize all blobs to the average height.

    Args:
        notes: Original (x, y, w, h) list.

    Returns:
        New list with heights = mean(h).
    """
    heights = [h for _, _, _, h in notes]
    avg_h = float(np.mean(heights))
    averaged: list[tuple[int,float,int,float]] = []
    for x, y, w, h in notes:
        cy = y + h / 2
        new_y = cy - avg_h / 2
        averaged.append((x, new_y, w, avg_h))
    return averaged  # type: ignore

def adjust_blob_height(
    notes: list[tuple[int,int,int,int]],
    height_factor: float
) -> list[tuple[int,float,int,float]]:
    """Scale all blobs’ height between min and max by factor.

    Args:
        notes: Original (x, y, w, h).
        height_factor: 0.0→1.0 scale between min_h and max_h.

    Returns:
        Adjusted (x, y, w, h_new).
    """
    hs = [h for _, _, _, h in notes]
    h_min, h_max = float(min(hs)), float(max(hs))
    new_h = h_min + (h_max - h_min) * height_factor
    adjusted: list[tuple[int,float,int,float]] = []
    for x, y, w, h in notes:
        cy = y + h / 2
        new_y = cy - new_h / 2
        adjusted.append((x, new_y, w, new_h))
    return adjusted  # type: ignore

def quantize_notes(
    notes: list[tuple[int,float,int,float]],
    lines: np.ndarray
) -> list[tuple[int,float,int,float]]:
    """Snap each blob vertically to its nearest staff slot.

    Args:
        notes: (x, y, w, h).
        lines: Y-positions array.

    Returns:
        Quantized blobs list.
    """
    quantized: list[tuple[int,float,int,float]] = []
    for x, y, w, h in notes:
        cy = y + h / 2
        for i in range(len(lines) - 1):
            if lines[i] <= cy < lines[i + 1]:
                mid = (lines[i] + lines[i + 1]) / 2
                slot_h = lines[i + 1] - lines[i] - 1
                new_y = mid - slot_h / 2
                quantized.append((x, new_y, w, slot_h))
                break
    return quantized  # type: ignore

def calculate_fit_accuracy(
    notes: list[tuple[int,float,int,float]],
    lines: np.ndarray
) -> float:
    """Percent of blobs *not* overlapping any staff line.

    Args:
        notes: (x, y, w, h).
        lines: 1D array.

    Returns:
        Accuracy in [0.0,100.0].
    """
    total = len(notes)
    if total == 0:
        return 0.0
    overlaps = 0
    for _, y, _, h in notes:
        top, bot = y, y + h
        for line in lines:
            if top <= line <= bot:
                overlaps += 1
                break
    return (total - overlaps) / total * 100.0

def calculate_note_variation(
    notes: list[tuple[int,float,int,float]],
    lines: np.ndarray
) -> float:
    """Average std-dev of blob centers within each staff slot.

    Args:
        notes: (x, y, w, h).
        lines: Y-positions array.

    Returns:
        Mean of per-slot std devs; 0 if none.
    """
    groups: list[list[float]] = [[] for _ in range(len(lines) - 1)]
    for _, y, _, h in notes:
        cy = y + h / 2
        for i in range(len(lines) - 1):
            if lines[i] <= cy < lines[i + 1]:
                groups[i].append(cy)
                break
    vars: list[float] = [
        float(np.std(g)) for g in groups if len(g) > 1
    ]
    return float(np.mean(vars)) if vars else 0.0

def create_staff_visualization(
    shape: tuple[int,int],
    notes: list[tuple[int,float,int,float]],
    lines: np.ndarray
) -> np.ndarray:
    """Render a blank staff with black‐filled blobs.

    Args:
        shape: (height, width) of canvas.
        notes: (x, y, w, h).
        lines: Y-positions.

    Returns:
        H×W×3 RGB image.
    """
    h, w = shape
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    for x, y, bw, bh in notes:
        cv2.rectangle(
            canvas,
            (int(x), int(y)),
            (int(x + bw), int(y + bh)),
            (0, 0, 0),
            -1
        )
    for ly in lines:
        cv2.line(canvas, (0, int(ly)), (w, int(ly)), (255, 0, 0), 1)
    return canvas
