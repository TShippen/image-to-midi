"""
Visualization functions for the image-to-MIDI pipeline.

This module centralizes all visualization functions used throughout the pipeline,
providing a consistent interface for creating visual representations of each stage.
"""

import cv2
import numpy as np
import pypianoroll
from collections.abc import Sequence
from matplotlib.figure import Figure

from image_to_midi.models.core_models import NoteBox, MidiEvent
from image_to_midi.models.pipeline_models import (
    BinaryResult,
    DetectionResult,
    StaffResult,
    MidiResult,
)
from image_to_midi.models.visualization_models import VisualizationSet


# -----------------------------------------------------------------------------
# 1) If you already have a List[MidiEvent], convert it to a pypianoroll.Multitrack
# -----------------------------------------------------------------------------
def midi_events_to_multitrack(
    events: Sequence[MidiEvent],
    resolution: int = 480,
    tempo: int = 120,
) -> pypianoroll.Multitrack:
    """Convert MidiEvent list to pypianoroll.Multitrack for visualization.

    Args:
        events: Sequence of MidiEvent objects.
        resolution: Ticks per quarter note (default 480).
        tempo: BPM for tempo array (default 120).

    Returns:
        pypianoroll.Multitrack object ready for plotting.
    """
    if not events:
        # Return empty Multitrack if no events
        return pypianoroll.Multitrack(resolution=resolution)

    # 1) Determine the last tick needed
    max_tick = max(e.start_tick + e.duration_tick for e in events)

    # 2) Allocate a (max_tick × 128) array of zeros
    pianoroll = np.zeros((max_tick, 128), dtype=np.uint8)

    # 3) For each MidiEvent, fill velocity=127 from start_tick to end_tick
    for event in events:
        if 0 <= event.note < 128:
            start = event.start_tick
            end = start + event.duration_tick
            pianoroll[start:end, event.note] = 127

    # 4) Build a single StandardTrack
    track = pypianoroll.StandardTrack(
        name="Generated Track",
        program=0,     # Acoustic Grand Piano
        is_drum=False, # pitched notes
        pianoroll=pianoroll,
    )

    # 5) Create a constant‐tempo array of length = max_tick
    tempo_array = np.full((max_tick,), tempo, dtype=np.float64)

    # 6) Wrap into a Multitrack
    return pypianoroll.Multitrack(
        resolution=resolution,
        tempo=tempo_array,
        tracks=[track],
    )


def create_piano_roll_from_events(
    events: Sequence[MidiEvent],
    *,
    width_px: int = 1200,
    note_h_in: float = 0.28,
    max_h_in: float = 12.0,
    min_h_in: float = 2.0,
    dpi: int = 150,
) -> Figure:
    """
    Given a list of MidiEvent, return a matplotlib Figure of the piano roll.

    Args:
        events: Sequence of MidiEvent (with .note, .start_tick, .duration_tick).
        width_px: Logical bitmap width (in pixels). In inches: width_px / dpi.
        note_h_in: Height (in inches) per semitone row.
        max_h_in: Maximum figure height, inches.
        min_h_in: Minimum figure height, inches.
        dpi: Raster DPI for the figure.

    Returns:
        A matplotlib Figure, ready for embedding (e.g. in Gradio).
    """
    import matplotlib.pyplot as plt

    # ---------- Empty‐case placeholder ----------
    if not events:
        fig, ax = plt.subplots(figsize=(width_px / dpi, min_h_in), dpi=dpi)
        ax.text(
            0.5,
            0.5,
            "No MIDI events",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return fig

    try:
        # Convert to Multitrack
        multitrack = midi_events_to_multitrack(events)

        # Compute pitch span from events
        lo_note = min(e.note for e in events)
        hi_note = max(e.note for e in events)
        pitch_span = hi_note - lo_note + 1

        width_in = width_px / dpi
        height_in = max(min_h_in, min(max_h_in, pitch_span * note_h_in))

        # Let pypianoroll create its own subplots with our size/dpi
        for _ax in multitrack.plot(ax=None, figsize=(width_in, height_in), dpi=dpi):
            pass

        # Grab the figure that pypianoroll created
        fig = plt.gcf()
        return fig

    except Exception as e:
        # Fall back to a simple “error” image
        fig, ax = plt.subplots(figsize=(width_px / dpi, min_h_in), dpi=dpi)
        ax.text(
            0.5,
            0.5,
            f"Error creating piano roll:\n{e}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            wrap=True,
        )
        ax.axis("off")
        return fig


# -----------------------------------------------------------------------------
# 2) New: boxes_to_multitrack → convert NoteBox + staff lines directly into a Multitrack
# -----------------------------------------------------------------------------
def boxes_to_multitrack(
    boxes: Sequence[NoteBox],
    lines: np.ndarray,
    base_midi: int = 60,
    *,
    ticks_per_px: int = 4,
    resolution: int = 480,
    tempo: int = 120,
) -> pypianoroll.Multitrack:
    """
    Convert a list of quantized NoteBox objects directly into a pypianoroll.Multitrack.

    Args:
        boxes:      Sequence of NoteBox (already snapped to staff lines).
        lines:      1D array of staff-line Y positions (length = n_lines, top→bottom).
        base_midi:  MIDI number for the bottom staff line (e.g. 60 = C4).
        ticks_per_px: How many MIDI ticks per horizontal pixel.
        resolution: MIDI ticks per quarter note (for the Multitrack header).
        tempo:      Constant BPM for the tempo track.

    Returns:
        A pypianoroll.Multitrack containing exactly one StandardTrack.
    """
    if not boxes or lines is None or lines.size == 0:
        # No boxes or no lines means “empty” Multitrack
        return pypianoroll.Multitrack(resolution=resolution)

    # 1) Build a mini‐list of (pitch, start_tick, duration_tick)
    events: list[tuple[int, int, int]] = []
    n_lines = lines.size

    for b in boxes:
        # Vertical center of the box
        center_y = b.y + b.h / 2.0

        # Find nearest staff‐line index (0 = top, n_lines-1 = bottom)
        idx = int(np.argmin(np.abs(lines - center_y)))

        # Convert line index into a MIDI pitch: bottom line = base_midi
        # (lines are top→bottom, so invert)
        pitch = base_midi + (n_lines - 1 - idx)

        # Convert horizontal coordinates into ticks
        start_tick = int(b.x * ticks_per_px)
        duration_tick = max(int(b.w * ticks_per_px), 1)

        events.append((pitch, start_tick, duration_tick))

    # 2) Figure out how many ticks we need total
    max_tick = max(start + dur for (_pitch, start, dur) in events)
    n_rows = max_tick + 1  # +1 so that an event ending at max_tick still fits

    # 3) Allocate a (n_rows × 128) uint8 array, all zeros
    pianoroll = np.zeros((n_rows, 128), dtype=np.uint8)

    # 4) Fill in each note’s ticks with velocity=127
    for pitch, start_tick, duration_tick in events:
        if 0 <= pitch < 128:
            end_tick = start_tick + duration_tick
            pianoroll[start_tick:end_tick, pitch] = 127

    # 5) Wrap into one StandardTrack
    track = pypianoroll.StandardTrack(
        name="BoxTrack",
        program=0,      # Acoustic Grand Piano
        is_drum=False,  # pitched notes
        pianoroll=pianoroll,
    )

    # 6) Build a constant‐tempo array of length = n_rows
    tempo_array = np.full((n_rows,), tempo, dtype=np.float64)

    # 7) Return the Multitrack
    return pypianoroll.Multitrack(
        resolution=resolution,
        tempo=tempo_array,
        tracks=[track],
    )


def create_piano_roll_from_boxes(
    boxes: Sequence[NoteBox],
    lines: np.ndarray,
    base_midi: int = 60,
    *,
    ticks_per_px: int = 4,
    resolution: int = 480,
    tempo: int = 120,
    width_px: int = 1200,
    note_h_in: float = 0.28,
    max_h_in: float = 12.0,
    min_h_in: float = 2.0,
    dpi: int = 150,
) -> Figure:
    """
    Generate a piano-roll Figure from quantized NoteBox objects (no MidiEvent needed).

    Args:
        boxes:       Sequence of NoteBox (quantized to staff lines).
        lines:       1D array of Y positions for staff lines (top→bottom).
        base_midi:   MIDI pitch for the bottom staff line (e.g. 60 = C4).
        ticks_per_px: MIDI ticks per horizontal pixel.
        resolution:  MIDI ticks per quarter note.
        tempo:       Constant BPM.
        width_px:    Logical bitmap width (in pixels).
        note_h_in:   Height (in inches) per semitone row.
        max_h_in:    Maximum total figure height, inches.
        min_h_in:    Minimum total figure height, inches.
        dpi:         Raster DPI for the figure.

    Returns:
        A matplotlib Figure of the piano roll.
    """
    import matplotlib.pyplot as plt

    # If there are no boxes or no staff lines, show a “No notes to plot” placeholder
    if not boxes or lines is None or lines.size == 0:
        fig, ax = plt.subplots(figsize=(width_px / dpi, min_h_in), dpi=dpi)
        ax.text(
            0.5,
            0.5,
            "No notes to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")
        return fig

    # 1) Convert boxes → Multitrack
    multitrack = boxes_to_multitrack(
        boxes=boxes,
        lines=lines,
        base_midi=base_midi,
        ticks_per_px=ticks_per_px,
        resolution=resolution,
        tempo=tempo,
    )

    # 2) Compute pitch span (min/max pitch from the boxes) to size the figure
    lo_pitch = min(
        base_midi
        + (lines.size - int(np.argmin(np.abs(lines - (b.y + b.h / 2.0)))) - 1)
        for b in boxes
    )
    hi_pitch = max(
        base_midi
        + (lines.size - int(np.argmin(np.abs(lines - (b.y + b.h / 2.0)))) - 1)
        for b in boxes
    )
    pitch_span = hi_pitch - lo_pitch + 1

    width_in = width_px / dpi
    height_in = max(min_h_in, min(max_h_in, pitch_span * note_h_in))

    # 3) Let pypianoroll plot it (it will create its own subplots)
    for _ax in multitrack.plot(ax=None, figsize=(width_in, height_in), dpi=dpi):
        pass

    # 4) Grab the figure that matplotlib is currently holding
    fig = plt.gcf()
    return fig


# -----------------------------------------------------------------------------
# 3) Everything else stays the same (no changes needed to these helpers)
# -----------------------------------------------------------------------------
def create_binary_visualization(binary_mask: np.ndarray | None) -> np.ndarray | None:
    """Create an RGB view of a binary mask, or None if no mask given."""
    if binary_mask is None:
        return None
    return cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)


def create_note_detection_visualizations(
    original_image: np.ndarray | None,
    binary_mask: np.ndarray | None,
    boxes: Sequence[NoteBox | tuple[int, int, int, int]],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Overlay detection boxes on both the color image and its mask.

    Args:
        original_image: H×W×3 BGR array, or None.
        binary_mask:    H×W mask, or None.
        boxes:          List of NoteBox or (x,y,w,h) tuples.

    Returns:
        (RGB-overlay, mask-overlay), either may be None if inputs are invalid.
    """
    if original_image is None or binary_mask is None or not boxes:
        return None, None

    # Prepare mask‐RGB and copies
    bin_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    rgb_viz = original_image.copy()
    bin_viz = bin_rgb.copy()

    for box in boxes:
        if hasattr(box, "x"):
            x, y, w, h = box.x, int(box.y), box.w, int(box.h)
        else:
            x, y, w, h = box  # assume a 4‐tuple

        cv2.rectangle(rgb_viz, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(bin_viz, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convert BGR→RGB for display
    rgb_viz = cv2.cvtColor(rgb_viz, cv2.COLOR_BGR2RGB)
    return rgb_viz, bin_viz


def create_staff_visualization(
    image_shape: tuple[int, int],
    notes: Sequence[NoteBox],
    lines: np.ndarray | None,
    fill_boxes: bool = True,
) -> np.ndarray | None:
    """
    Draw staff lines and (optionally filled) note boxes on a blank canvas.

    Args:
        image_shape: (height, width) in pixels.
        notes:       Sequence of NoteBox objects.
        lines:       1D array of Y-positions, or None.
        fill_boxes:  If True, draw solid rectangles; else just outlines.

    Returns:
        H×W×3 uint8 image, or None if `notes` is empty or `lines` is None/empty.
    """
    if not notes or lines is None or lines.size == 0:
        return None

    h, w = image_shape
    canvas = np.full((h, w, 3), 255, np.uint8)

    # Draw filled or outlined rectangles for each note
    for n in notes:
        color = (0, 0, 0)
        thickness = -1 if fill_boxes else 2
        cv2.rectangle(
            canvas,
            (int(n.x), int(n.y)),
            (int(n.x + n.w), int(n.y + n.h)),
            color,
            thickness,
        )

    # Draw staff lines in red
    for ly in lines:
        cv2.line(canvas, (0, int(ly)), (w, int(ly)), (255, 0, 0), 1)

    return canvas


def create_detection_visualizations(
    image: np.ndarray | None,
    binary_result: BinaryResult | None,
    detection_result: DetectionResult | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    High‐level “pipeline” wrapper around note detection visualizations.

    Args:
        image:            RGB image, or None.
        binary_result:    may have `.binary_mask`, or None.
        detection_result: may have `.note_boxes`, or None.

    Returns:
        (RGB overlay, mask overlay) or (None, None) on invalid inputs.
    """
    if (
        image is None
        or binary_result is None
        or binary_result.binary_mask is None
        or detection_result is None
        or not detection_result.note_boxes
    ):
        return None, None

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return create_note_detection_visualizations(
        bgr,
        binary_result.binary_mask,
        detection_result.note_boxes,
    )


def create_staff_result_visualizations(
    image_shape: tuple[int, int],
    staff_result: StaffResult | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    High‐level wrapper for drawing original vs. quantized staff boxes.

    Args:
        image_shape:  (height, width) or None.
        staff_result: may have `.lines`, `.original_boxes`, `.quantized_boxes`, or None.

    Returns:
        (orig_viz, quant_viz) or (None, None) on invalid inputs.
    """
    if (
        staff_result is None
        or staff_result.lines is None
        or staff_result.lines.size == 0
    ):
        return None, None

    orig_viz = None
    if staff_result.original_boxes:
        orig_viz = create_staff_visualization(
            image_shape, staff_result.original_boxes, staff_result.lines
        )

    quant_viz = None
    if staff_result.quantized_boxes:
        quant_viz = create_staff_visualization(
            image_shape, staff_result.quantized_boxes, staff_result.lines
        )

    return orig_viz, quant_viz


def create_all_visualizations(
    image: np.ndarray | None,
    binary_result: BinaryResult | None,
    detection_result: DetectionResult | None,
    staff_result: StaffResult | None,
    midi_result: MidiResult | None,
) -> VisualizationSet:
    """
    Bundle everything into one VisualizationSet for the UI.

    Args:
        image:            RGB image, or None.
        binary_result:    may have `.binary_mask`, or None.
        detection_result: may have `.note_boxes`, or None.
        staff_result:     may have lines/boxes, or None.
        midi_result:      may have `.events`, or None.

    Returns:
        A VisualizationSet instance (fields inside it may be None).
    """
    if image is None:
        return VisualizationSet()

    # 1) Binary mask (just store it; UI might display it)
    binary_mask = binary_result.binary_mask if binary_result else None

    # 2) Note-detection overlays
    note_rgb, note_bin = create_detection_visualizations(
        image, binary_result, detection_result
    )

    # 3) Staff result overlays (original vs. quantized boxes)
    staff_rgb, staff_quant = create_staff_result_visualizations(
        image.shape[:2], staff_result
    )

    # 4) Piano-roll: prefer “events” path; fallback to “boxes” path
    piano_roll = None
    if midi_result and midi_result.events:
        piano_roll = create_piano_roll_from_events(
            midi_result.events, width_px=1400, dpi=180
        )
    else:
        if staff_result and staff_result.quantized_boxes and staff_result.lines is not None:
            piano_roll = create_piano_roll_from_boxes(
                boxes=staff_result.quantized_boxes,
                lines=staff_result.lines,
                base_midi=60,      # adjust if your base MIDI pitch differs
                ticks_per_px=4,    # must match build_note_events's ticks_per_px
                resolution=480,
                tempo=120,
                width_px=1400,
                dpi=180,
            )

    return VisualizationSet(
        binary_mask=binary_mask,
        note_detection=note_rgb,
        note_detection_binary=note_bin,
        staff_lines=staff_rgb,
        quantized_notes=staff_quant,
        piano_roll=piano_roll,
    )
