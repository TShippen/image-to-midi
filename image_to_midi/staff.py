import cv2
import numpy as np

from image_to_midi.models import NoteBox


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _rescale_height(notes: list[NoteBox], target_h: float) -> list[NoteBox]:
    """Return new NoteBox list where every box height == target_h."""
    out: list[NoteBox] = []
    half = target_h / 2
    for n in notes:
        out.append(
            n.model_copy(
                update={
                    "y": n.cy - half,
                    "h": target_h,
                }
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Public functions
# --------------------------------------------------------------------------- #
def detect_lines(notes: list[NoteBox], num_lines: int) -> np.ndarray:
    """Equally spaced staff-line Y positions spanning the note centres."""
    centres = [n.cy for n in notes]
    return np.linspace(min(centres), max(centres), num_lines)


def average_box_height(notes: list[NoteBox]) -> list[NoteBox]:
    """Replace every box height with the global mean height."""
    avg = float(np.mean([n.h for n in notes]))
    return _rescale_height(notes, avg)


def adjust_box_height(notes: list[NoteBox], factor: float) -> list[NoteBox]:
    """Scale all heights between min and max by *factor* ∈ [0,1]."""
    h_min, h_max = min(n.h for n in notes), max(n.h for n in notes)
    target = h_min + (h_max - h_min) * factor
    return _rescale_height(notes, float(target))


def vertical_quantize_notes(notes: list[NoteBox], lines: np.ndarray) -> list[NoteBox]:
    """Snap each NoteBox vertically into its nearest staff slot."""
    out: list[NoteBox] = []
    for n in notes:
        for i in range(len(lines) - 1):
            if lines[i] <= n.cy < lines[i + 1]:
                slot_h = float(lines[i + 1] - lines[i] - 1)
                out.append(
                    n.model_copy(
                        update={
                            "y": (lines[i] + lines[i + 1]) / 2 - slot_h / 2,
                            "h": slot_h,
                        }
                    )
                )
                break
    return out


def calculate_fit_accuracy(notes: list[NoteBox], lines: np.ndarray) -> float:
    """Percent of boxes *not* overlapping any staff line."""
    total = len(notes)
    if total == 0:
        return 0.0

    overlaps = 0
    for n in notes:
        top, bot = n.y, n.y + n.h
        if any(top <= ly <= bot for ly in lines):
            overlaps += 1
    return (total - overlaps) / total * 100.0


def calculate_note_variation(notes: list[NoteBox], lines: np.ndarray) -> float:
    """Mean std-dev of vertical centres within each staff slot."""
    groups: list[list[float]] = [[] for _ in range(len(lines) - 1)]
    for n in notes:
        for i in range(len(lines) - 1):
            if lines[i] <= n.cy < lines[i + 1]:
                groups[i].append(n.cy)
                break

    deviations = [float(np.std(g)) for g in groups if len(g) > 1]
    return float(np.mean(deviations)) if deviations else 0.0
