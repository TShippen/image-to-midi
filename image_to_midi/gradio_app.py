"""
Image-to-MIDI conversion app with Gradio interface.

This application provides an interactive interface for converting paint splatter images
to MIDI music. It visualizes the entire pipeline from image processing to MIDI generation,
allowing users to see how parameter changes affect the final output.
"""

import gradio as gr
import numpy as np

from image_to_midi.pipeline import process_complete_pipeline
from image_to_midi.models import (
    ImageProcessingParams,
    NoteDetectionParams,
    StaffParams,
    MidiParams,
)


def update_pipeline_ui(
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
    tempo_bpm,
):
    """Update the entire pipeline based on all parameters.

    Args:
        image: Input image data
        threshold: Binarization threshold
        min_area: Minimum blob area
        max_area: Maximum blob area
        min_aspect: Minimum aspect ratio
        max_aspect: Maximum aspect ratio
        method: Staff line fitting method
        num_lines: Number of staff lines
        height_factor: Height adjustment factor
        base_midi: Base MIDI note
        tempo_bpm: Tempo in BPM

    Returns:
        All outputs required by the Gradio interface
    """
    if image is None:
        return [None] * 12  # Return empty placeholders for all outputs

    # Create parameter models
    image_params = ImageProcessingParams(threshold=threshold)
    detection_params = NoteDetectionParams(
        min_area=min_area,
        max_area=max_area,
        min_aspect_ratio=min_aspect,
        max_aspect_ratio=max_aspect,
    )
    staff_params = StaffParams(
        method=method,
        num_lines=num_lines,
        height_factor=height_factor,
    )
    midi_params = MidiParams(
        base_midi_note=base_midi,
        tempo_bpm=tempo_bpm,
    )

    # Process the complete pipeline
    _, detection_result, staff_result, midi_result, vis_set = process_complete_pipeline(
        image, image_params, detection_params, staff_params, midi_params
    )

    # Format outputs for Gradio interface
    return (
        vis_set.binary_mask,
        vis_set.note_detection,
        vis_set.note_detection_binary,
        vis_set.staff_lines,
        vis_set.quantized_notes,
        vis_set.piano_roll,
        f"{staff_result.fit_accuracy:.2f}%",
        f"{staff_result.pitch_variation:.2f}",
        f"{len(detection_result.note_boxes)} notes detected",
        f"Base note: {midi_result.base_note_name}",
        midi_result.midi_file_path if midi_result.midi_file_path else None,
        midi_result.midi_bytes if midi_result.midi_bytes else None,
    )


def create_gradio_interface():
    """Create and configure the Gradio interface for the Image-to-MIDI app.

    Returns:
        gr.Blocks: Configured Gradio interface
    """
    with gr.Blocks(title="Image to MIDI Converter") as interface:
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
                with gr.Group():
                    gr.Markdown("### 1. Image Processing")
                    gr.Markdown(
                        "Control how the image is converted to a binary mask that identifies potential notes."
                    )
                    threshold = gr.Slider(
                        minimum=0,
                        maximum=255,
                        value=93,
                        step=1,
                        label="Threshold",
                        info="Controls how light/dark areas become notes. Higher values detect lighter areas.",
                    )

                # 2. Note Detection parameters
                with gr.Group():
                    gr.Markdown("### 2. Note Detection")
                    gr.Markdown(
                        "Filter shapes in the binary image to identify those that represent notes."
                    )
                    min_area = gr.Slider(
                        0.01,
                        10.0,
                        1.0,
                        0.01,
                        label="Min Area",
                        info="Minimum area (in pixels) for a detected shape to be considered a note",
                    )
                    max_area = gr.Slider(
                        100,
                        5000,
                        5000,
                        100,
                        label="Max Area",
                        info="Maximum area (in pixels) for a detected shape to be considered a note",
                    )
                    min_aspect = gr.Slider(
                        0.1,
                        5.0,
                        0.1,
                        0.01,
                        label="Min Aspect Ratio",
                        info="Minimum width/height ratio for a shape to be considered a note",
                    )
                    max_aspect = gr.Slider(
                        5.0,
                        50.0,
                        20.0,
                        0.5,
                        label="Max Aspect Ratio",
                        info="Maximum width/height ratio for a shape to be considered a note",
                    )

                # 3. Staff Line parameters
                with gr.Group():
                    gr.Markdown("### 3. Staff Line Fitting")
                    gr.Markdown(
                        "Create musical staff lines and align the detected notes to them."
                    )
                    method = gr.Radio(
                        ["Original", "Average", "Adjustable"],
                        value="Original",
                        label="Line Fitting Method",
                        info="How to adjust note heights for staff fitting",
                    )
                    num_lines = gr.Slider(
                        2,
                        50,
                        10,
                        1,
                        label="Number of Lines",
                        info="Number of staff lines to create (more lines = more notes)",
                    )
                    height_factor = gr.Slider(
                        0.0,
                        1.0,
                        0.5,
                        0.01,
                        label="Height Factor (for Adjustable method)",
                        info="Controls note height adjustment when using Adjustable method",
                    )

                # 4. MIDI Generation parameters
                with gr.Group():
                    gr.Markdown("### 4. MIDI Generation")
                    gr.Markdown(
                        "Convert the staff notes to MIDI events and generate a playable file."
                    )
                    base_midi = gr.Slider(
                        21,
                        108,
                        60,
                        1,
                        label="Base MIDI Note",
                        info="The MIDI note number for the lowest staff line",
                    )
                    tempo_bpm = gr.Slider(
                        30,
                        240,
                        120,
                        1,
                        label="Tempo (BPM)",
                        info="Speed of playback in beats per minute",
                    )

                # Metrics display
                with gr.Group():
                    gr.Markdown("### Analysis Metrics")
                    gr.Markdown(
                        "Statistical metrics about the notes and staff line fit."
                    )
                    with gr.Row():
                        accuracy_display = gr.Textbox(label="Fit Accuracy")
                        variation_display = gr.Textbox(label="Pitch Variation")
                    notes_count = gr.Textbox(label="Detected Notes")
                    base_note_display = gr.Textbox(label="Base Note")

            # Right column - Visualizations & Output
            with gr.Column(scale=1):
                # Process visualizations in sequential order
                with gr.Group():
                    gr.Markdown("### Stage 1: Binary Image")
                    gr.Markdown(
                        "The image is converted to a binary mask where potential notes are white."
                    )
                    binary_output = gr.Image(label="Binary Mask")

                with gr.Group():
                    gr.Markdown("### Stage 2: Note Detection")
                    gr.Markdown(
                        "Shapes in the binary image are filtered by size and aspect ratio to identify notes."
                    )
                    with gr.Row():
                        note_vis_rgb = gr.Image(label="Notes on Original")
                        note_vis_bin = gr.Image(label="Notes on Binary")

                with gr.Group():
                    gr.Markdown("### Stage 3: Staff Line Fitting")
                    gr.Markdown(
                        "Staff lines are created and notes are quantized to align with the lines."
                    )
                    with gr.Row():
                        staff_viz = gr.Image(label="Staff Lines")
                        quant_viz = gr.Image(label="Quantized Notes")

                with gr.Group():
                    gr.Markdown("### Stage 4: MIDI Output")
                    gr.Markdown(
                        "Notes are converted to MIDI events and visualized as a piano roll."
                    )
                    piano_roll = gr.Image(label="Piano Roll")
                    midi_audio = gr.Audio(label="MIDI Playback", type="filepath")
                    midi_download = gr.File(label="Download MIDI")

        # Connect all parameters to update the entire pipeline
        input_controls = [
            input_image,
            threshold,
            min_area,
            max_area,
            min_aspect,
            max_aspect,
            method,
            num_lines,
            height_factor,
            base_midi,
            tempo_bpm,
        ]

        output_components = [
            binary_output,
            note_vis_rgb,
            note_vis_bin,
            staff_viz,
            quant_viz,
            piano_roll,
            accuracy_display,
            variation_display,
            notes_count,
            base_note_display,
            midi_audio,
            midi_download,
        ]

        # Connect all inputs to the update function
        for control in input_controls:
            control.change(
                fn=update_pipeline_ui, inputs=input_controls, outputs=output_components
            )

    return interface


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
