"""
Visualization functions for the image-to-MIDI pipeline.

This module centralizes all visualization functions used throughout the pipeline,
providing a consistent interface for creating visual representations of each stage.
"""

import cv2
import numpy as np
from collections.abc import Sequence

from image_to_midi.models.core_models import NoteBox, MidiEvent
from image_to_midi.models.pipeline_models import (
    BinaryResult,
    DetectionResult,
    StaffResult,
    MidiResult,
)
from image_to_midi.models.visualization_models import VisualizationSet


def create_binary_visualization(binary_mask: np.ndarray | None) -> np.ndarray | None:
    """Create an RGB view of a binary mask, or None if no mask given.

    Args:
        binary_mask: 2D uint8 array, or None.

    Returns:
        3-channel uint8 array, or None if input was None.
    """
    if binary_mask is None:
        return None
    return cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)


def create_note_detection_visualizations(
    original_image: np.ndarray | None,
    binary_mask: np.ndarray | None,
    boxes: Sequence[NoteBox | tuple[int, int, int, int]],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Overlay detection boxes on both the color image and its mask.

    Args:
        original_image: H×W×3 BGR array, or None.
        binary_mask:   H×W mask, or None.
        boxes:         List of NoteBox or (x,y,w,h) tuples.

    Returns:
        (RGB-overlay, mask-overlay), either may be None if inputs are invalid.
    """
    if original_image is None or binary_mask is None or not boxes:
        return None, None

    # prepare mask‐RGB
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

    # convert BGR→RGB for display
    rgb_viz = cv2.cvtColor(rgb_viz, cv2.COLOR_BGR2RGB)
    return rgb_viz, bin_viz


def create_staff_visualization(
    image_shape: tuple[int, int],
    notes: Sequence[NoteBox],
    lines: np.ndarray | None,
    fill_boxes: bool = True,
) -> np.ndarray | None:
    """Draw staff lines and (optionally filled) note boxes on a blank canvas.

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

    # draw notes
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

    # draw lines
    for ly in lines:
        cv2.line(canvas, (0, int(ly)), (w, int(ly)), (255, 0, 0), 1)

    return canvas


def create_piano_roll_visualization(
    events,
    *,
    width_px: int = 1200,
    note_h_in: float = 0.28,
    max_h_in: float = 12.0,
    min_h_in: float = 2.0,
    dpi: int = 150,
    margin_frac: float = 0.05          # white gap above & below each bar
):
    """
    Return a Matplotlib piano-roll figure ready for display in Gradio.

    A bar now sits fully between the black guide-lines:

        ─────── guide line at pitch-0.5
          ███   bar (pitch row interior)
        ─────── guide line at pitch+0.5

    Parameters
    ----------
    events        : Sequence[MidiEvent]  (needs .note, .start_tick, .duration_tick)
    width_px      : logical bitmap width; Gradio will scale the <img>.
    note_h_in     : physical height of one pitch row when few rows are present.
    max_h_in/min_h_in : clamp total figure height, inches.
    dpi           : raster DPI.
    margin_frac   : fraction of a row left clear at top & bottom of every bar.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import hsv_to_rgb

    # ---------- empty case ----------
    if not events:
        fig, ax = plt.subplots(figsize=(width_px / dpi, min_h_in), dpi=dpi)
        ax.text(0.5, 0.5, "No MIDI events", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    # ---------- basic extents ----------
    lo_note  = min(e.note for e in events)
    hi_note  = max(e.note for e in events)
    lo_tick  = min(e.start_tick for e in events)
    hi_tick  = max(e.start_tick + e.duration_tick for e in events)

    pitch_span = hi_note - lo_note + 1          # number of rows

    # ---------- figure size ----------
    width_in  = width_px / dpi
    height_in = max(min_h_in, min(max_h_in, pitch_span * note_h_in))
    fig, ax   = plt.subplots(figsize=(width_in, height_in), dpi=dpi)

    # ---------- axes limits ----------
    ax.set_xlim(lo_tick, hi_tick)
    ax.set_ylim(lo_note - 0.5, hi_note + 0.5)   # rows are centred on ints

    # ---------- guide lines at row boundaries (half-integers) ----------
    for k in range(pitch_span + 1):             # one extra for top border
        y = lo_note - 0.5 + k                   # 59.5, 60.5, …
        ax.axhline(y, color="black", linewidth=1, zorder=0)

    # ---------- draw note rectangles fully inside each row ----------
    cell_h   = 1 - 2 * margin_frac             # height of coloured bar
    y_shift  = 0.5 - margin_frac               # distance from pitch number
                                               #   to bar *top*
    for ev in events:
        rgb = hsv_to_rgb(((ev.note % 12) / 12.0, 0.8, 0.85))
        ax.add_patch(
            patches.Rectangle(
                (ev.start_tick, ev.note - y_shift),     # bottom-left
                ev.duration_tick,                       # width
                cell_h,                                 # height
                facecolor=rgb,
                edgecolor="black",
                linewidth=0.8,
                zorder=1
            )
        )

    # ---------- cosmetics ----------
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")
    fig.tight_layout()

    return fig



def create_detection_visualizations(
    image: np.ndarray | None,
    binary_result: BinaryResult | None,
    detection_result: DetectionResult | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """High‐level “pipeline” wrapper around note detection visualizations.

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
    """High‐level wrapper for drawing original vs. quantized staff boxes.

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
    """Bundle everything into one VisualizationSet for the UI.

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

    binary_mask = binary_result.binary_mask if binary_result else None
    note_rgb, note_bin = create_detection_visualizations(
        image, binary_result, detection_result
    )
    staff_rgb, staff_quant = create_staff_result_visualizations(
        image.shape[:2], staff_result
    )

    piano_roll = None
    if midi_result and midi_result.events:
        piano_roll = create_piano_roll_visualization(midi_result.events, width_px=1400, dpi=180)

    return VisualizationSet(
        binary_mask=binary_mask,
        note_detection=note_rgb,
        note_detection_binary=note_bin,
        staff_lines=staff_rgb,
        quantized_notes=staff_quant,
        piano_roll=piano_roll,
    )
