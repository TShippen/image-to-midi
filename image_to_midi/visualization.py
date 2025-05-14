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


def create_binary_visualization(binary_mask: np.ndarray) -> np.ndarray | None:
    """Create a visualization of a binary mask.
    
    Args:
        binary_mask: Binary image as NumPy array
        
    Returns:
        RGB visualization of the binary mask or None if input is None
    """
    if binary_mask is None:
        return None
        
    # Convert binary to RGB for display
    return cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)


def create_note_detection_visualizations(
    original_image: np.ndarray,
    binary_mask: np.ndarray,
    boxes: Sequence
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Create visualizations of detected notes on both original and binary images.
    
    This function draws bounding boxes on copies of the original image and binary mask.
    It does not duplicate detection logic, but simply visualizes the results.
    
    Args:
        original_image: Original BGR image
        binary_mask: Binary mask image
        boxes: List of (x, y, w, h) tuples or NoteBox objects
        
    Returns:
        Tuple of (RGB visualization, binary visualization), either may be None if inputs are invalid
    """
    if original_image is None or binary_mask is None or not boxes:
        return None, None
    
    # Convert binary to RGB for drawing colored rectangles
    bin_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    
    # Create copies of the images to draw on
    rgb_viz = original_image.copy()
    bin_viz = bin_rgb.copy()
    
    # Draw rectangles on both visualizations
    for box in boxes:
        if hasattr(box, 'x'):  # It's a NoteBox
            x, y, w, h = box.x, int(box.y), box.w, int(box.h)
        else:  # It's a tuple
            x, y, w, h = box
            
        cv2.rectangle(rgb_viz, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(bin_viz, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Convert BGR to RGB for display
    rgb_viz = cv2.cvtColor(rgb_viz, cv2.COLOR_BGR2RGB)
    
    return rgb_viz, bin_viz


def create_staff_visualization(
    image_shape: tuple,
    notes: Sequence[NoteBox],
    lines: np.ndarray,
    fill_boxes: bool = True
) -> np.ndarray | None:
    """Render a visualization of notes and staff lines.
    
    Args:
        image_shape: Tuple of (height, width) for the canvas
        notes: List of NoteBox objects to draw
        lines: NumPy array of staff line y-positions
        fill_boxes: Whether to fill the note boxes (True) or just draw outlines (False)
        
    Returns:
        RGB visualization with notes and staff lines or None if inputs are invalid
    """
    if not notes or lines is None or len(lines) == 0:
        return None
    
    height, width = image_shape
    canvas = np.ones((height, width, 3), np.uint8) * 255
    
    # Draw notes
    for n in notes:
        color = (0, 0, 0)  # Black notes
        thickness = -1 if fill_boxes else 2  # Filled or outline
        
        cv2.rectangle(
            canvas,
            (int(n.x), int(n.y)),
            (int(n.x + n.w), int(n.y + n.h)),
            color,
            thickness,
        )
    
    # Draw staff lines
    for ly in lines:
        cv2.line(canvas, (0, int(ly)), (width, int(ly)), (255, 0, 0), 1)
    
    return canvas


def create_piano_roll_visualization(events: Sequence[MidiEvent]) -> np.ndarray:
    """Create piano roll visualization for MIDI events.
    
    Args:
        events: List of MidiEvent objects
        
    Returns:
        RGB visualization of the piano roll (always returns a valid image)
    """
    if not events:
        # Return empty canvas if no events
        return np.ones((20, 200, 3), np.uint8) * 255
    
    # Calculate time and pitch bounds
    end_tick = max(e.start_tick + e.duration_tick for e in events)
    lowest = min(e.note for e in events)
    highest = max(e.note for e in events)
    
    pitch_span = highest - lowest + 1
    height = pitch_span * 10  # 10 px per semitone
    width = int(end_tick * 1.1)  # 10% right-hand margin
    
    # Create canvas
    img = np.ones((height, width, 3), np.uint8) * 255
    
    # Draw horizontal key-lines (white keys darker)
    for i in range(pitch_span + 1):
        y = i * 10
        key_color = (
            (150, 150, 150)
            if (lowest + i) % 12 in [0, 2, 4, 5, 7, 9, 11]
            else (200, 200, 200)
        )
        cv2.line(img, (0, y), (width, y), key_color, 1)
    
    # Draw each note block
    for ev in events:
        y0 = (ev.note - lowest) * 10
        top, bot = height - y0 - 10, height - y0
        left, right = ev.start_tick, ev.start_tick + ev.duration_tick
        
        # Get color based on note pitch (hue from note value)
        hsv = np.array([[[((ev.note % 12) / 12) * 180, 200, 200]]], np.uint8)
        fill = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0].tolist()
        
        # Draw filled note rectangle
        cv2.rectangle(img, (left, top), (right, bot), fill, -1)
        
        # Draw note outline
        cv2.rectangle(img, (left, top), (right, bot), (0, 0, 0), 1)
    
    return img


# Pipeline model visualization functions

def create_detection_visualizations(
    image: np.ndarray,
    binary_result: BinaryResult,
    detection_result: DetectionResult
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Create RGB and binary visualizations of detected notes.
    
    Args:
        image: Original RGB image
        binary_result: BinaryResult with binary mask
        detection_result: DetectionResult with note boxes
        
    Returns:
        Tuple of (RGB visualization, binary visualization), either may be None if inputs are invalid
    """
    if (image is None or binary_result is None or 
            binary_result.binary_mask is None or 
            detection_result is None or not detection_result.note_boxes):
        return None, None
    
    # Convert to BGR for OpenCV processing
    bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create visualizations using the detected note boxes
    return create_note_detection_visualizations(
        bgr_img, 
        binary_result.binary_mask, 
        detection_result.note_boxes
    )


def create_staff_result_visualizations(
    image_shape: tuple,
    staff_result: StaffResult
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Create visualizations for both original and quantized staff notes.
    
    Args:
        image_shape: Original image dimensions (height, width)
        staff_result: StaffResult with lines and note boxes
        
    Returns:
        Tuple of (original visualization, quantized visualization), either may be None
    """
    if (staff_result is None or 
            staff_result.lines is None or staff_result.lines.size == 0):
        return None, None
    
    # Create original notes visualization if available
    orig_viz = None
    if staff_result.original_boxes:
        orig_viz = create_staff_visualization(
            image_shape, 
            staff_result.original_boxes, 
            staff_result.lines
        )
    
    # Create quantized notes visualization if available
    quant_viz = None
    if staff_result.quantized_boxes:
        quant_viz = create_staff_visualization(
            image_shape, 
            staff_result.quantized_boxes, 
            staff_result.lines
        )
    
    return orig_viz, quant_viz


def create_all_visualizations(
    image: np.ndarray,
    binary_result: BinaryResult,
    detection_result: DetectionResult,
    staff_result: StaffResult,
    midi_result: MidiResult
) -> VisualizationSet:
    """Create a complete set of visualizations for the pipeline.
    
    Args:
        image: Original RGB image
        binary_result: Binary processing result
        detection_result: Note detection result
        staff_result: Staff creation result
        midi_result: MIDI generation result
        
    Returns:
        Complete VisualizationSet for UI (all fields may be None if visualization fails)
    """
    if image is None:
        return VisualizationSet()
    
    # Get binary mask
    binary_mask = binary_result.binary_mask if binary_result else None
    
    # Create detection visualizations
    note_vis_rgb, note_vis_bin = create_detection_visualizations(
        image, binary_result, detection_result
    )
    
    # Create staff visualizations
    img_shape = image.shape[:2]
    staff_viz, quant_viz = create_staff_result_visualizations(
        img_shape, staff_result
    )
    
    # Create piano roll visualization
    piano_roll = None
    if midi_result and midi_result.events:
        piano_roll = create_piano_roll_visualization(midi_result.events)
    
    # Package into visualization set
    return VisualizationSet(
        binary_mask=binary_mask,
        note_detection=note_vis_rgb,
        note_detection_binary=note_vis_bin,
        staff_lines=staff_viz,
        quantized_notes=quant_viz,
        piano_roll=piano_roll
    )
