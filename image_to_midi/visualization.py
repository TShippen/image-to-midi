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


def create_piano_roll_visualization(events: Sequence[MidiEvent]) -> np.ndarray:
    """Render a piano‐roll (time vs. pitch) image for a list of events.
    Args:
        events: List of MidiEvent (note, start_tick, duration_tick).
    Returns:
        Always returns a valid H×W×3 uint8 array (white background).
    """
    if not events:
        return np.full((20, 200, 3), 255, np.uint8)

    # Calculate actual start and end ticks to trim white space
    start_tick = min(e.start_tick for e in events)
    end_tick = max(e.start_tick + e.duration_tick for e in events)
    tick_span = max(1, end_tick - start_tick)

    lowest = min(e.note for e in events)
    highest = max(e.note for e in events)
    pitch_span = max(1, highest - lowest + 1)

    # Add extra padding at top and bottom
    padding_notes = 2  # Add 2 extra note spaces at top and bottom
    display_pitch_span = pitch_span + (padding_notes * 2)

    # Target total height and width
    target_height = 500  # Target total image height
    target_width = 1000  # Target total image width

    # Calculate height per note as a ratio of available space
    height_per_note = target_height / display_pitch_span

    # Calculate the actual image dimensions
    height = int(display_pitch_span * height_per_note)
    width = int(target_width)

    # Scale the tick span to fit the width
    horizontal_padding = 40  # 20px padding on each side
    tick_to_pixel_ratio = (width - horizontal_padding) / tick_span

    # Create the image
    img = np.full((height, width, 3), 255, np.uint8)

    # Adjust the vertical positions to account for padding
    display_lowest = lowest - padding_notes
    display_highest = highest + padding_notes

    # horizontal key lines
    for i in range(display_pitch_span + 1):
        y = int(i * height_per_note)
        note_value = display_lowest + i
        is_white = note_value % 12 in [0, 2, 4, 5, 7, 9, 11]
        key_color = (150, 150, 150) if is_white else (200, 200, 200)
        cv2.line(img, (0, y), (width, y), key_color, 1)

    # draw notes
    for ev in events:
        # Calculate note position with padding offset
        note_idx = ev.note - display_lowest
        y0 = int(note_idx * height_per_note)
        # Adjust to stay within the lines
        top = height - y0 - int(height_per_note) + 1  # +1 to stay inside the line
        bot = height - y0 - 1  # -1 to stay inside the line

        # Calculate x position based on scaled tick span
        left = int(horizontal_padding // 2 + (ev.start_tick - start_tick) * tick_to_pixel_ratio)
        right = int(horizontal_padding // 2 + (ev.start_tick + ev.duration_tick - start_tick) * tick_to_pixel_ratio)

        # Ensure minimum note width of 2px
        if right - left < 2:
            right = left + 2

        hsv = np.array([[[((ev.note % 12) / 12) * 180, 200, 200]]], np.uint8)
        fill = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].tolist()
        cv2.rectangle(img, (left, top), (right, bot), fill, -1)
        cv2.rectangle(img, (left, top), (right, bot), (0, 0, 0), 1)

    return img

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
        piano_roll = create_piano_roll_visualization(midi_result.events)

    return VisualizationSet(
        binary_mask=binary_mask,
        note_detection=note_rgb,
        note_detection_binary=note_bin,
        staff_lines=staff_rgb,
        quantized_notes=staff_quant,
        piano_roll=piano_roll,
    )
