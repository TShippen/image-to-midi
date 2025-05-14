"""
Visualization functions for the image-to-MIDI pipeline.

This module centralizes all visualization functions used throughout the pipeline,
providing a consistent interface for creating visual representations of each stage.
"""

import cv2
import numpy as np

from image_to_midi.models.core_models import NoteBox
from image_to_midi.models.pipeline_models import (
    BinaryResult,
    DetectionResult,
    StaffResult,
    MidiResult,
)
from image_to_midi.models.visualization_models import VisualizationSet


def create_binary_visualization(binary_mask):
    """Create a visualization of a binary mask.
    
    Args:
        binary_mask: Binary image as NumPy array
        
    Returns:
        RGB visualization of the binary mask
    """
    if binary_mask is None:
        return None
        
    # Convert binary to RGB for display
    return cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)


def create_note_detection_visualization(original_image, binary_mask, note_boxes):
    """Create visualizations of detected notes on both original and binary images.
    
    Args:
        original_image: Original BGR or RGB image
        binary_mask: Binary mask image
        note_boxes: List of raw box tuples (x, y, w, h) or NoteBox objects
        
    Returns:
        Tuple of (RGB visualization, binary visualization)
    """
    if original_image is None or binary_mask is None or not note_boxes:
        return None, None
    
    # Ensure image is in BGR format for OpenCV
    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
        bgr_img = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR) if original_image.shape[2] == 3 else original_image
    else:
        # Handle grayscale case
        bgr_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    # Convert binary to RGB for drawing colored rectangles
    bin_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    
    # Convert NoteBox objects to tuples if needed
    if note_boxes and hasattr(note_boxes[0], 'x'):
        # These are NoteBox objects
        box_tuples = [(box.x, int(box.y), box.w, int(box.h)) for box in note_boxes]
    else:
        # These are already tuples
        box_tuples = note_boxes
    
    # Clone images to avoid modifying originals
    rgb_viz = bgr_img.copy()
    bin_viz = bin_rgb.copy()
    
    # Draw rectangles on both visualizations
    for x, y, w, h in box_tuples:
        cv2.rectangle(rgb_viz, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(bin_viz, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Ensure output is RGB for display
    rgb_viz = cv2.cvtColor(rgb_viz, cv2.COLOR_BGR2RGB)
    
    return rgb_viz, bin_viz


def create_staff_visualization(image_shape, note_boxes, staff_lines, fill_boxes=True):
    """Render a visualization of notes and staff lines.
    
    Args:
        image_shape: Tuple of (height, width) for the canvas
        note_boxes: List of NoteBox objects to draw
        staff_lines: NumPy array of staff line y-positions
        fill_boxes: Whether to fill the note boxes (True) or just draw outlines (False)
        
    Returns:
        RGB visualization with notes and staff lines
    """
    if not note_boxes or staff_lines is None or len(staff_lines) == 0:
        return None
    
    height, width = image_shape
    canvas = np.ones((height, width, 3), np.uint8) * 255
    
    # Draw notes
    for n in note_boxes:
        if fill_boxes:
            # Filled rectangle for notes
            cv2.rectangle(
                canvas,
                (int(n.x), int(n.y)),
                (int(n.x + n.w), int(n.y + n.h)),
                (0, 0, 0),
                -1,
            )
        else:
            # Outlined rectangle for notes
            cv2.rectangle(
                canvas,
                (int(n.x), int(n.y)),
                (int(n.x + n.w), int(n.y + n.h)),
                (0, 0, 0),
                2,
            )
    
    # Draw staff lines
    for ly in staff_lines:
        cv2.line(canvas, (0, int(ly)), (width, int(ly)), (255, 0, 0), 1)
    
    return canvas


def create_piano_roll_visualization(midi_events):
    """Create piano roll visualization for MIDI events.
    
    Args:
        midi_events: List of MidiEvent objects
        
    Returns:
        RGB visualization of the piano roll
    """
    if not midi_events:
        # Return empty canvas if no events
        return np.ones((20, 200, 3), np.uint8) * 255
    
    # Calculate time and pitch bounds
    end_tick = max(e.start_tick + e.duration_tick for e in midi_events)
    lowest = min(e.note for e in midi_events)
    highest = max(e.note for e in midi_events)
    
    pitch_span = highest - lowest + 1
    height = pitch_span * 10  # 10 px per semitone
    width = int(end_tick * 1.1)  # 10% right-hand margin
    
    # Create canvas
    img = np.ones((height, width, 3), np.uint8) * 255
    
    # Draw horizontal key-lines (white keys darker)
    for i in range(pitch_span + 1):
        y = i * 10
        key_colour = (
            (150, 150, 150)
            if (lowest + i) % 12 in [0, 2, 4, 5, 7, 9, 11]
            else (200, 200, 200)
        )
        cv2.line(img, (0, y), (width, y), key_colour, 1)
    
    # Draw each note block
    for ev in midi_events:
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


def create_detection_rgb_visualization(image, binary_result, detection_result):
    """Create RGB visualization of detected notes on the original image.
    
    Args:
        image: Original RGB image
        binary_result: BinaryResult with binary mask
        detection_result: DetectionResult with note boxes
        
    Returns:
        RGB visualization with boxes drawn on original image or None if error occurs
    """
    if (image is None or binary_result is None or 
            binary_result.binary_mask is None or 
            detection_result is None or not detection_result.note_boxes):
        return None
    
    # Convert to BGR for OpenCV processing
    bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create visualization
    vis_rgb, _ = create_note_detection_visualization(
        bgr_img, 
        binary_result.binary_mask, 
        detection_result.note_boxes
    )
    
    return vis_rgb


def create_detection_binary_visualization(image, binary_result, detection_result):
    """Create binary visualization of detected notes.
    
    Args:
        image: Original RGB image
        binary_result: BinaryResult with binary mask
        detection_result: DetectionResult with note boxes
        
    Returns:
        Binary visualization with boxes drawn on binary mask or None if error occurs
    """
    if (image is None or binary_result is None or 
            binary_result.binary_mask is None or 
            detection_result is None or not detection_result.note_boxes):
        return None
    
    # Convert to BGR for OpenCV processing
    bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create visualization
    _, vis_bin = create_note_detection_visualization(
        bgr_img, 
        binary_result.binary_mask, 
        detection_result.note_boxes
    )
    
    return vis_bin


def create_staff_original_visualization(image_shape, staff_result):
    """Create visualization for original staff lines and notes.
    
    Args:
        image_shape: Original image dimensions (height, width)
        staff_result: StaffResult with lines and note boxes
        
    Returns:
        Visualization of original notes with staff lines or None if error occurs
    """
    if (staff_result is None or not staff_result.original_boxes or 
            staff_result.lines is None or staff_result.lines.size == 0):
        return None
    
    # Create visualization
    return create_staff_visualization(
        image_shape, 
        staff_result.original_boxes, 
        staff_result.lines
    )


def create_staff_quantized_visualization(image_shape, staff_result):
    """Create visualization for quantized notes on staff lines.
    
    Args:
        image_shape: Original image dimensions (height, width)
        staff_result: StaffResult with lines and quantized note boxes
        
    Returns:
        Visualization of quantized notes with staff lines or None if error occurs
    """
    if (staff_result is None or not staff_result.quantized_boxes or 
            staff_result.lines is None or staff_result.lines.size == 0):
        return None
    
    # Create visualization
    return create_staff_visualization(
        image_shape, 
        staff_result.quantized_boxes, 
        staff_result.lines
    )


def create_piano_roll_from_midi_result(midi_result):
    """Create piano roll visualization from a MidiResult object.
    
    Args:
        midi_result: MidiResult with MIDI events
        
    Returns:
        Piano roll image as numpy array or None if error occurs
    """
    if midi_result is None or not midi_result.events:
        return None
    
    # Create piano roll
    return create_piano_roll_visualization(midi_result.events)


def create_visualization_set(image, binary_result, detection_result, staff_result, midi_result):
    """Create a complete set of visualizations for the pipeline.
    
    Args:
        image: Original RGB image
        binary_result: Binary processing result
        detection_result: Note detection result
        staff_result: Staff creation result
        midi_result: MIDI generation result
        
    Returns:
        Complete VisualizationSet for UI
    """
    if image is None:
        return VisualizationSet()
    
    # Generate binary mask visualization
    binary_mask = binary_result.binary_mask if binary_result else None
    
    # Generate note detection visualizations
    note_vis_rgb = create_detection_rgb_visualization(
        image, binary_result, detection_result
    )
    
    note_vis_bin = create_detection_binary_visualization(
        image, binary_result, detection_result
    )
    
    # Generate staff visualizations
    img_shape = image.shape[:2]
    staff_vis = create_staff_original_visualization(img_shape, staff_result)
    quant_vis = create_staff_quantized_visualization(img_shape, staff_result)
    
    # Generate piano roll visualization
    piano_roll = create_piano_roll_from_midi_result(midi_result)
    
    # Package into visualization set
    return VisualizationSet(
        binary_mask=binary_mask,
        note_detection=note_vis_rgb,
        note_detection_binary=note_vis_bin,
        staff_lines=staff_vis,
        quantized_notes=quant_vis,
        piano_roll=piano_roll
    )
