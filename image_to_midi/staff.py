import numpy as np
from image_to_midi.models import NoteBox


def _rescale_height(notes: list[NoteBox], target_h: float) -> list[NoteBox]:
    """Rescale all NoteBox heights to a fixed target and re-center vertically.

    Args:
        notes: List of NoteBox instances.
        target_h: Desired height for each box.

    Returns:
        A new list of NoteBox objects with `h == target_h` and
        `y` adjusted so each box remains centered at its original midpoint.
        Returns an empty list if `notes` is empty.
    """
    if not notes:
        return []
    half = target_h / 2.0
    return [
        n.model_copy(update={"y": n.y + n.h / 2.0 - half, "h": target_h}) for n in notes
    ]


def detect_lines(notes: list[NoteBox], num_lines: int) -> np.ndarray:
    """Compute equally spaced staff‐line Y positions spanning the note tops.

    Args:
        notes: List of NoteBox instances.
        num_lines: Number of staff lines to generate.

    Returns:
        A 1D numpy array of length `num_lines`, from min(y) to max(y).
        Returns an empty array if `notes` is empty or `num_lines <= 0`.
    """
    if not notes or num_lines <= 0:
        return np.array([])
    tops = [n.y for n in notes]
    return np.linspace(min(tops), max(tops), num_lines)


def average_box_height(notes: list[NoteBox]) -> list[NoteBox]:
    """Normalize all box heights to their global mean height.

    Args:
        notes: List of NoteBox instances.

    Returns:
        A new list of NoteBox objects with every height set to the
        mean of the original heights. Returns an empty list if `notes` is empty.
    """
    if not notes:
        return []
    mean_h = float(np.mean([n.h for n in notes]))
    return _rescale_height(notes, mean_h)


def adjust_box_height(notes: list[NoteBox], factor: float) -> list[NoteBox]:
    """Scale all box heights between min and max by a given factor.

    Args:
        notes: List of NoteBox instances.
        factor: A float in [0, 1]:
            - 0.0 → all boxes set to the minimum height
            - 1.0 → all boxes set to the maximum height

    Returns:
        A new list of NoteBox objects with heights interpolated between
        the original min and max by `factor`. Returns an empty list if `notes` is empty.
    """
    if not notes:
        return []
    heights = [n.h for n in notes]
    target_h = min(heights) + (max(heights) - min(heights)) * factor
    return _rescale_height(notes, target_h)


def vertical_quantize_notes(notes: list[NoteBox], lines: np.ndarray) -> list[NoteBox]:
    """Snap each NoteBox vertically into its nearest staff slot.

    Args:
        notes: List of NoteBox instances.
        lines: 1D numpy array of staff‐line Y positions; must have length ≥ 2.

    Returns:
        A new list of NoteBox objects with `y` and `h` updated so that each
        box sits between the two closest `lines`. If `notes` is empty or
        `lines` has fewer than 2 entries, returns an empty list.
    """
    if not notes or lines.size < 2:
        return []
    result: list[NoteBox] = []
    for n in notes:
        center = n.y + n.h / 2.0
        for i in range(lines.size - 1):
            low, high = lines[i], lines[i + 1]
            if low <= center < high:
                slot_h = float(high - low - 1)
                new_y = (low + high) / 2.0 - slot_h / 2.0
                result.append(n.model_copy(update={"y": new_y, "h": slot_h}))
                break
    return result


def horizontal_quantize_notes(
    notes: list[NoteBox],
    image_width: int,
    grid_divisions: int = 16,
    strength: float = 1.0,
) -> list[NoteBox]:
    """Quantize NoteBox horizontal positions and widths to an equal grid.

    Args:
        notes: List of NoteBox instances.
        image_width: Width of the image in pixels.
        grid_divisions: Number of equal divisions (must be > 0).
        strength: Quantization strength [0.0 = none, 1.0 = full snap].

    Returns:
        A new list of NoteBox objects with `x` and `w` moved toward the
        nearest grid lines by `strength`. If `notes` is empty,
        or `grid_divisions <= 0`, or `strength <= 0`, returns `notes` unchanged.
    """
    if not notes or grid_divisions <= 0 or strength <= 0:
        return notes

    positions = np.linspace(0.0, float(image_width), grid_divisions + 1)
    result: list[NoteBox] = []

    for n in notes:
        left, right = float(n.x), float(n.x + n.w)
        idx_l = int(np.argmin(np.abs(positions - left)))
        idx_r = int(np.argmin(np.abs(positions - right)))
        ql, qr = positions[idx_l], positions[idx_r]

        new_left = left * (1.0 - strength) + ql * strength
        new_right = right * (1.0 - strength) + qr * strength
        new_w = max(1, int(new_right - new_left))

        result.append(n.model_copy(update={"x": int(new_left), "w": new_w}))

    return result


def calculate_fit_accuracy(notes: list[NoteBox], lines: np.ndarray) -> float:
    """Compute the percentage of boxes not overlapping any staff line.

    Args:
        notes: List of NoteBox instances.
        lines: 1D numpy array of staff‐line Y positions.

    Returns:
        A float in [0.0, 100.0], the percentage of NoteBoxes whose
        vertical span does not include any line. Returns 0.0 if `notes`
        is empty. If `lines` is empty, returns 100.0.
    """
    total = len(notes)
    if total == 0:
        return 0.0
    if lines.size == 0:
        return 100.0

    overlaps = 0
    for n in notes:
        top, bottom = n.y, n.y + n.h
        if any(top <= ly <= bottom for ly in lines):
            overlaps += 1

    return (total - overlaps) / total * 100.0


def calculate_note_variation(notes: list[NoteBox], lines: np.ndarray) -> float:
    """Compute mean standard deviation of vertical centers within each staff slot.

    Args:
        notes: List of NoteBox instances.
        lines: 1D numpy array of staff‐line Y positions; length ≥ 2.

    Returns:
        The mean of standard deviations of box-center `y` within each slot
        (between consecutive lines). Returns 0.0 if there are fewer than
        two lines or no slot has more than one note.
    """
    if not notes or lines.size < 2:
        return 0.0

    slots: list[list[float]] = [[] for _ in range(lines.size - 1)]
    for n in notes:
        center = n.y + n.h / 2.0
        for i in range(lines.size - 1):
            if lines[i] <= center < lines[i + 1]:
                slots[i].append(center)
                break

    deviations = [float(np.std(group)) for group in slots if len(group) > 1]
    return float(np.mean(deviations)) if deviations else 0.0
