"""
Image-to-MIDI conversion app with Gradio interface.

This application provides an interactive interface for converting paint splatter images
to MIDI music. It visualizes the entire pipeline from image processing to MIDI generation,
allowing users to see how parameter changes affect the final output.
"""

import os
import tempfile
import gradio as gr
import cv2
import numpy as np
from PIL import Image
from collections.abc import Sequence

from image_to_midi.image_processing import preprocess_image, create_note_visualization
from image_to_midi.note_detection import detect_notes
from image_to_midi.staff import (
    detect_lines,
    average_box_height,
    adjust_box_height,
    quantize_notes,
    calculate_fit_accuracy,
    calculate_note_variation,
    create_staff_visualization,
)
from image_to_midi.midi_utils import (
    build_note_events,
    write_midi_file,
    create_piano_roll,
)
from image_to_midi.models import NoteBox, MidiEvent


def process_image(image: np.ndarray, threshold_value: int) -> tuple[np.ndarray, np.ndarray]:
    """Process the uploaded image to create a binary mask.
    
    Args:
        image: RGB image as a NumPy array.
        threshold_value: Threshold value for binarization (0-255).
        
    Returns:
        tuple: (original RGB image, binary mask image)
    """
    if image is None:
        return None, None
    
    # Convert from RGB to BGR for OpenCV
    bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Process image
    binary_mask = preprocess_image(bgr_img, threshold_value)
    
    return image, binary_mask


def detect_note_boxes(
    image: np.ndarray, 
    binary_mask: np.ndarray, 
    min_area: float, 
    max_area: float, 
    min_aspect: float, 
    max_aspect: float
) -> tuple[np.ndarray, np.ndarray, list]:
    """Detect note-like shapes in the binary image.
    
    Args:
        image: Original RGB image.
        binary_mask: Binary mask from preprocessing.
        min_area: Minimum contour area to consider.
        max_area: Maximum contour area to consider.
        min_aspect: Minimum aspect ratio (width/height) to consider.
        max_aspect: Maximum aspect ratio (width/height) to consider.
        
    Returns:
        tuple: (visualization on RGB, visualization on binary, list of NoteBox objects)
    """
    if image is None or binary_mask is None:
        return None, None, []
    
    # Convert back to BGR for processing
    bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Detect notes
    raw_boxes = detect_notes(binary_mask, min_area, max_area, min_aspect, max_aspect)
    
    # Convert to NoteBox models
    note_boxes = [
        NoteBox(x=x, y=float(y), w=w, h=float(h)) for x, y, w, h in raw_boxes
    ]
    
    # Create visualizations
    vis_rgb, vis_bin = create_note_visualization(bgr_img, binary_mask, raw_boxes)
    
    # Convert BGR back to RGB for display
    vis_rgb = cv2.cvtColor(vis_rgb, cv2.COLOR_BGR2RGB)
    
    return vis_rgb, vis_bin, note_boxes


def fit_staff_lines(
    image_shape: tuple, 
    note_boxes: list, 
    method: str, 
    num_lines: int, 
    height_factor: float
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """Fit staff lines to detected notes and quantize notes to lines.
    
    Args:
        image_shape: Original image dimensions (height, width).
        note_boxes: List of NoteBox objects.
        method: Line fitting method ('Original', 'Average', or 'Adjustable').
        num_lines: Number of staff lines to create.
        height_factor: Height adjustment factor (0-1) for 'Adjustable' method.
        
    Returns:
        tuple: (staff visualization, quantized visualization, quantized notes, staff lines)
    """
    if not note_boxes or image_shape is None:
        return None, None, [], np.array([])
    
    # Choose working list of NoteBoxes based on method
    if method == "Original":
        working_boxes = note_boxes
    elif method == "Average":
        working_boxes = average_box_height(note_boxes)
    else:  # Adjustable
        working_boxes = adjust_box_height(note_boxes, height_factor)
    
    # Create staff lines
    lines = detect_lines(working_boxes, num_lines)
    
    # Quantize notes to staff lines
    quantized = quantize_notes(working_boxes, lines)
    
    # Create visualizations
    viz_orig = create_staff_visualization(image_shape, working_boxes, lines)
    viz_quant = create_staff_visualization(image_shape, quantized, lines)
    
    return viz_orig, viz_quant, quantized, lines


def create_midi(
    quantized_boxes: list, 
    lines: np.ndarray, 
    base_midi: int, 
    tempo_bpm: int
) -> tuple[np.ndarray, str, bytes]:
    """Generate MIDI from quantized notes.
    
    Args:
        quantized_boxes: List of quantized NoteBox objects.
        lines: Array of staff line y-positions.
        base_midi: Base MIDI note number.
        tempo_bpm: Tempo in beats per minute.
        
    Returns:
        tuple: (piano roll visualization, midi file path, midi data bytes)
    """
    if not quantized_boxes or lines.size == 0:
        return None, "", None
    
    # Build MIDI events
    events = build_note_events(quantized_boxes, lines, base_midi)
    
    # Create piano roll visualization
    roll_img = create_piano_roll(events)
    
    # Generate MIDI file
    midi_bytes = write_midi_file(events, tempo_bpm)
    
    # Save MIDI to a temporary file for playback
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
        tmp.write(midi_bytes)
        midi_path = tmp.name
    
    return roll_img, midi_path, midi_bytes


def calculate_metrics(note_boxes: list, lines: np.ndarray) -> tuple[float, float]:
    """Calculate quality metrics for staff line fitting.
    
    Args:
        note_boxes: List of NoteBox objects.
        lines: Array of staff line y-positions.
        
    Returns:
        tuple: (fit accuracy percentage, pitch variation score)
    """
    if not note_boxes or lines.size == 0:
        return 0.0, 0.0
    
    accuracy = calculate_fit_accuracy(note_boxes, lines)
    variation = calculate_note_variation(note_boxes, lines)
    
    return accuracy, variation


def get_midi_note_name(midi_note: int) -> str:
    """Convert MIDI note number to note name with octave.
    
    Args:
        midi_note: MIDI note number (0-127).
        
    Returns:
        str: Note name with octave (e.g., "C4").
    """
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note = note_names[midi_note % 12]
    octave = (midi_note // 12) - 1
    return f"{note}{octave}"


def update_pipeline(
    image, 
    threshold, 
    min_area, 
    max_area, 
    min_aspect, 
    max_aspect, 
    method, 
    num_lines, 
    height_factor, 
    base_midi, 
    tempo_bpm
):
    """Update the entire pipeline based on all parameters.
    
    This function processes an image through the complete pipeline:
    image → binary → note detection → staff fitting → MIDI generation.
    
    Args:
        image: Input image data.
        threshold: Binarization threshold.
        min_area: Minimum blob area.
        max_area: Maximum blob area.
        min_aspect: Minimum aspect ratio.
        max_aspect: Maximum aspect ratio.
        method: Staff line fitting method.
        num_lines: Number of staff lines.
        height_factor: Height adjustment factor.
        base_midi: Base MIDI note.
        tempo_bpm: Tempo in BPM.
        
    Returns:
        All outputs required by the Gradio interface.
    """
    if image is None:
        return [None] * 12  # Return empty placeholders for all outputs
    
    # Step 1: Process image
    _, binary_mask = process_image(image, threshold)
    
    # Step 2: Detect notes
    note_vis_rgb, note_vis_bin, note_boxes = detect_note_boxes(
        image, binary_mask, min_area, max_area, min_aspect, max_aspect
    )
    
    # Get image dimensions for staff visualization
    img_h, img_w = image.shape[:2] if image is not None else (0, 0)
    
    # Step 3: Fit staff lines
    staff_viz, quant_viz, quantized_boxes, lines = fit_staff_lines(
        (img_h, img_w), note_boxes, method, num_lines, height_factor
    )
    
    # Calculate metrics
    accuracy, variation = calculate_metrics(note_boxes, lines)
    
    # Step 4: Generate MIDI
    piano_roll, midi_path, midi_bytes = create_midi(
        quantized_boxes, lines, base_midi, tempo_bpm
    )
    
    # Get note name for display
    base_note_name = get_midi_note_name(base_midi)
    
    # Count detections
    num_notes = len(note_boxes)
    
    return (
        binary_mask,
        note_vis_rgb,
        note_vis_bin,
        staff_viz,
        quant_viz,
        piano_roll,
        f"{accuracy:.2f}%",
        f"{variation:.2f}",
        f"{num_notes} notes detected",
        f"Base note: {base_note_name}",
        midi_path if midi_path else None,
        midi_bytes if midi_bytes else None
    )


def create_gradio_interface():
    """Create and configure the Gradio interface for the Image-to-MIDI app.
    
    Returns:
        gr.Blocks: Configured Gradio interface.
    """
    with gr.Blocks(title="Image to MIDI Converter") as demo:
        gr.Markdown("# 🎵 Image to MIDI Converter")
        gr.Markdown(
            "Upload a paint splatter image and convert it to music! "
            "See how different parameter choices affect the entire process."
        )
        
        # Main layout
        with gr.Row():
            # Left column - Parameters
            with gr.Column(scale=1):
                # Image upload
                input_image = gr.Image(label="Upload Image", type="numpy")
                
                # 1. Image Processing parameters
                with gr.Box():
                    gr.Markdown("### 1. Image Processing")
                    threshold = gr.Slider(
                        minimum=0, 
                        maximum=255, 
                        value=93, 
                        step=1, 
                        label="Threshold"
                    )
                
                # 2. Note Detection parameters
                with gr.Box():
                    gr.Markdown("### 2. Note Detection")
                    min_area = gr.Slider(0.01, 10.0, 1.0, 0.01, label="Min Area")
                    max_area = gr.Slider(100, 5000, 5000, 100, label="Max Area")
                    min_aspect = gr.Slider(0.1, 5.0, 0.1, 0.01, label="Min Aspect Ratio")
                    max_aspect = gr.Slider(5.0, 50.0, 20.0, 0.5, label="Max Aspect Ratio")
                
                # 3. Staff Line parameters
                with gr.Box():
                    gr.Markdown("### 3. Staff Line Fitting")
                    method = gr.Radio(
                        ["Original", "Average", "Adjustable"], 
                        value="Original", 
                        label="Line Fitting Method"
                    )
                    num_lines = gr.Slider(2, 50, 10, 1, label="Number of Lines")
                    height_factor = gr.Slider(
                        0.0, 1.0, 0.5, 0.01, 
                        label="Height Factor (for Adjustable method)"
                    )
                
                # 4. MIDI Generation parameters
                with gr.Box():
                    gr.Markdown("### 4. MIDI Generation")
                    base_midi = gr.Slider(21, 108, 60, 1, label="Base MIDI Note")
                    tempo_bpm = gr.Slider(30, 240, 120, 1, label="Tempo (BPM)")
                    
                # Metrics display
                with gr.Box():
                    gr.Markdown("### Analysis Metrics")
                    with gr.Row():
                        accuracy_display = gr.Textbox(label="Fit Accuracy")
                        variation_display = gr.Textbox(label="Pitch Variation")
                    notes_count = gr.Textbox(label="Detected Notes")
                    base_note_display = gr.Textbox(label="Base Note")
            
            # Right column - Visualizations & Output
            with gr.Column(scale=1):
                # Process visualizations in sequential order
                with gr.Box():
                    gr.Markdown("### Stage 1: Binary Image")
                    binary_output = gr.Image(label="Binary Mask")
                
                with gr.Box():
                    gr.Markdown("### Stage 2: Note Detection")
                    with gr.Row():
                        note_vis_rgb = gr.Image(label="Notes on Original")
                        note_vis_bin = gr.Image(label="Notes on Binary")
                
                with gr.Box():
                    gr.Markdown("### Stage 3: Staff Line Fitting")
                    with gr.Row():
                        staff_viz = gr.Image(label="Staff Lines")
                        quant_viz = gr.Image(label="Quantized Notes")
                
                with gr.Box():
                    gr.Markdown("### Stage 4: MIDI Output")
                    piano_roll = gr.Image(label="Piano Roll")
                    midi_audio = gr.Audio(label="MIDI Playback", type="filepath")
                    midi_download = gr.File(label="Download MIDI")
        
        # Connect all parameters to update the entire pipeline
        input_controls = [
            input_image, threshold, 
            min_area, max_area, min_aspect, max_aspect,
            method, num_lines, height_factor, 
            base_midi, tempo_bpm
        ]
        
        output_components = [
            binary_output, 
            note_vis_rgb, note_vis_bin, 
            staff_viz, quant_viz, 
            piano_roll,
            accuracy_display, variation_display, 
            notes_count, base_note_display,
            midi_audio, midi_download
        ]
        
        # Connect all inputs to the update function
        for control in input_controls:
            control.change(
                fn=update_pipeline,
                inputs=input_controls,
                outputs=output_components
            )
    
    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
