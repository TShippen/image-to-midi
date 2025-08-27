"""
Visualization functions for the image-to-MIDI pipeline.

This module centralizes all visualization functions used throughout the pipeline,
providing a consistent interface for creating visual representations of each stage.
"""

import cv2
import numpy as np
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


def create_piano_roll_visualization(
    events: Sequence[MidiEvent],
    *,
    width_px: int = 1200,
    note_h_in: float = 0.28,
    max_h_in: float = 12.0,
    min_h_in: float = 2.0,
    dpi: int = 150,
    margin_frac: float = 0.05,
) -> Figure:
    """Create a piano roll visualization of MIDI events with note labels.

    Generates a horizontal timeline chart showing MIDI notes as colored bars,
    with each row representing a different pitch. Notes are color-coded by
    pitch class and include note name labels on the y-axis.

    Args:
        events: Sequence of MIDI events to visualize.
        width_px: Logical bitmap width in pixels (default 1200).
        note_h_in: Physical height per pitch row in inches (default 0.28).
        max_h_in: Maximum figure height in inches (default 12.0).
        min_h_in: Minimum figure height in inches (default 2.0).
        dpi: Raster resolution for output (default 150).
        margin_frac: Fraction of row height to leave as margin (default 0.05).

    Returns:
        Matplotlib Figure object containing the piano roll visualization.
        Returns a figure with "No MIDI events" message if events is empty.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import hsv_to_rgb

    # ---------- empty case ----------
    if not events:
        fig, ax = plt.subplots(figsize=(width_px / dpi, min_h_in), dpi=dpi)
        ax.text(
            0.5, 0.5, "No MIDI events", ha="center", va="center", transform=ax.transAxes
        )
        ax.axis("off")
        return fig

    # ---------- basic extents ----------
    lo_note = min(e.note for e in events)
    hi_note = max(e.note for e in events)
    lo_tick = min(e.start_tick for e in events)
    hi_tick = max(e.start_tick + e.duration_tick for e in events)

    pitch_span = hi_note - lo_note + 1  # number of rows

    # ---------- figure size ----------
    width_in = width_px / dpi
    height_in = max(min_h_in, min(max_h_in, pitch_span * note_h_in))
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)

    # ---------- axes limits ----------
    ax.set_xlim(lo_tick, hi_tick)
    ax.set_ylim(lo_note - 0.5, hi_note + 0.5)  # rows are centred on ints

    # ---------- guide lines at row boundaries (half-integers) ----------
    for k in range(pitch_span + 1):  # one extra for top border
        y = lo_note - 0.5 + k  # 59.5, 60.5, …
        ax.axhline(y, color="black", linewidth=1, zorder=0)

    # ---------- draw note rectangles fully inside each row ----------
    cell_h = 1 - 2 * margin_frac  # height of coloured bar
    y_shift = 0.5 - margin_frac  # distance from pitch number
    #   to bar *top*
    for ev in events:
        rgb = hsv_to_rgb(((ev.note % 12) / 12.0, 0.8, 0.85))
        ax.add_patch(
            patches.Rectangle(
                (ev.start_tick, ev.note - y_shift),  # bottom-left
                ev.duration_tick,  # width
                cell_h,  # height
                facecolor=rgb,
                edgecolor="black",
                linewidth=0.8,
                zorder=1,
            )
        )

    # ---------- add note name labels on Y-axis ----------
    # Create note name labels for all visible notes
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    visible_notes = range(lo_note, hi_note + 1)
    y_ticks = []
    y_labels = []

    for note_num in visible_notes:
        y_ticks.append(note_num)
        note_name = note_names[note_num % 12]
        octave = (note_num // 12) - 1
        y_labels.append(f"{note_name}{octave}")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)

    # ---------- cosmetics ----------
    ax.set_xticks([])  # Hide X-axis ticks
    ax.set_xlabel("Time (ticks)", fontsize=10)
    ax.set_ylabel("Note", fontsize=10)

    # Keep the left spine for Y-axis labels, hide others
    for spine_name, spine in ax.spines.items():
        if spine_name != "left":
            spine.set_visible(False)

    ax.set_facecolor("white")
    fig.tight_layout()

    # Important: Ensure matplotlib closes the figure after Gradio uses it
    # This prevents memory leaks from accumulated figure objects
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    return fig


def create_binary_visualization(binary_mask: np.ndarray | None) -> np.ndarray | None:
    """Convert a binary mask to RGB format for display.

    Args:
        binary_mask: 2D binary image array, or None.

    Returns:
        3-channel RGB version of the binary mask, or None if input is None.
    """
    if binary_mask is None:
        return None
    return cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)


def create_note_detection_visualizations(
    original_image: np.ndarray | None,
    binary_mask: np.ndarray | None,
    boxes: Sequence[NoteBox | tuple[int, int, int, int]],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Overlay detection bounding boxes on original and binary images.

    Creates visualization images showing detected note boxes as red rectangles
    overlaid on both the original color image and the binary mask version.

    Args:
        original_image: Original BGR image as H×W×3 array, or None.
        binary_mask: Binary mask as H×W array, or None.
        boxes: Sequence of NoteBox objects or (x,y,w,h) tuples representing
               detected note bounding boxes.

    Returns:
        Tuple of (RGB_overlay, mask_overlay) where RGB_overlay shows boxes
        on the original image converted to RGB, and mask_overlay shows boxes
        on the binary mask. Either may be None if inputs are invalid.
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
    """Create a visualization showing staff lines and note boxes.

    Renders staff lines as red horizontal lines and note boxes as black
    rectangles (filled or outlined) on a white canvas. Used to visualize
    the staff line generation and note quantization results.

    Args:
        image_shape: Canvas dimensions as (height, width) in pixels.
        notes: Sequence of NoteBox objects to draw.
        lines: 1D array of staff line y-positions, or None.
        fill_boxes: If True, draw filled rectangles; if False, draw outlines only.

    Returns:
        RGB image as H×W×3 uint8 array showing the visualization, or None
        if notes is empty or lines is None/empty.
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
    """Create detection visualizations from pipeline results.

    High-level wrapper that extracts data from pipeline result objects
    and creates visualization images showing detected note boxes.

    Args:
        image: Input RGB image, or None.
        binary_result: Binary processing result containing mask, or None.
        detection_result: Note detection result containing boxes, or None.

    Returns:
        Tuple of (RGB_overlay, mask_overlay) showing detected boxes on
        original and binary images, or (None, None) if inputs are invalid.
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
    """Create visualizations of original and quantized staff results.

    High-level wrapper that generates two visualizations: one showing
    original detected notes with staff lines, and another showing the
    same staff lines with quantized note positions.

    Args:
        image_shape: Canvas dimensions as (height, width) in pixels.
        staff_result: Staff processing result containing lines and note boxes, or None.

    Returns:
        Tuple of (original_viz, quantized_viz) showing staff lines with
        original and quantized notes respectively, or (None, None) if
        staff_result is invalid.
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
    """Create complete set of visualizations from all pipeline results.

    Master function that generates all visualization types from the outputs
    of each pipeline stage, bundling them into a single VisualizationSet
    object for easy consumption by the user interface.

    Args:
        image: Input RGB image, or None.
        binary_result: Binary processing result, or None.
        detection_result: Note detection result, or None.
        staff_result: Staff processing result, or None.
        midi_result: MIDI generation result, or None.

    Returns:
        VisualizationSet containing all generated visualizations. Individual
        fields may be None if corresponding inputs were invalid or processing failed.
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

    # 4) Piano-roll visualization
    piano_roll = None
    if midi_result and midi_result.events:
        piano_roll = create_piano_roll_visualization(midi_result.events)

    return VisualizationSet(
        binary_mask=binary_mask,
        note_detection=note_rgb,
        note_detection_binary=note_bin,
        staff_lines=staff_rgb,
        quantized_notes=staff_quant,
        piano_roll=piano_roll,
    )
