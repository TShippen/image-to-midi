"""
Image-to-MIDI conversion app with Gradio interface.
This application provides an interactive interface for converting paint splatter images
to MIDI music. It visualizes the entire pipeline from image processing to MIDI generation,
allowing users to see how parameter changes affect the final output.
"""

import logging
import gradio as gr
import cv2
import tempfile
from image_to_midi.pipeline import process_complete_pipeline
from image_to_midi.models import (
    ImageProcessingParams,
    NoteDetectionParams,
    StaffParams,
    MidiParams,
)
from image_to_midi.music_transformations import (
    get_available_key_signatures,
    get_available_scale_names,
    NOTE_VALUE_TO_GRID,
)

logger = logging.getLogger(__name__)

# Load the fixed image once at the start
FIXED_IMAGE = cv2.imread("img/paint_splatter.png")
# Convert from BGR to RGB for Gradio
if FIXED_IMAGE is not None:
    FIXED_IMAGE = cv2.cvtColor(FIXED_IMAGE, cv2.COLOR_BGR2RGB)


def update_pipeline_ui(
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
    fit_to_scale,
    root_note,
    scale_type,
    quantize,
    note_value,
):
    """Update the entire pipeline based on all parameters."""
    # Use the fixed image instead of an uploaded one
    image = FIXED_IMAGE

    if image is None:
        logger.error("Could not load the fixed image at img/paint_splatter.png")
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

    # Convert note_value to grid_size
    grid_size = NOTE_VALUE_TO_GRID[note_value]

    midi_params = MidiParams(
        base_midi_note=base_midi,
        tempo_bpm=tempo_bpm,
        # New fields for music transformations
        map_to_scale=fit_to_scale,
        scale_key=root_note,
        scale_type=scale_type.lower(),
        quantize_rhythm=quantize,
        grid_size=grid_size,
        quantize_strength=1.0,
    )

    # Process the complete pipeline
    binary_result, detection_result, staff_result, midi_result, vis_set = (
        process_complete_pipeline(
            image, image_params, detection_params, staff_params, midi_params
        )
    )

    # Create a temp file for the downloadable MIDI if it doesn't exist
    midi_download_path = None
    if midi_result.midi_bytes:
        # Create a temporary file with .mid extension for download
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp.write(midi_result.midi_bytes)
            midi_download_path = tmp.name

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
        midi_download_path,  # Use the temporary path for the download link
    )


def create_gradio_interface():
    """Create and configure the Gradio interface for the Image-to-MIDI app."""
    with gr.Blocks(title="Image to MIDI Converter") as interface:
        gr.Markdown("# 🎵 Image to MIDI Converter")
        gr.Markdown(
            "This app converts a paint splatter image to music! "
            "Adjust the parameters to see how they affect the entire process."
        )

        # Use a parallel layout with two main columns
        with gr.Row():
            # Left column - Parameters
            with gr.Column(scale=1):
                # Show the fixed image at a reasonable size
                gr.Image(
                    value="img/paint_splatter.png",
                    label="Source Image",
                    interactive=False,
                    height=200,
                )

                # 1. Image Processing parameters with immediate visualization
                with gr.Group():
                    gr.Markdown("### 1. Image Processing")
                    with gr.Row():
                        # Left side - controls
                        with gr.Column(scale=1):
                            gr.Markdown(
                                "Control how the image is converted to a binary mask that identifies potential notes."
                            )
                            threshold = gr.Slider(
                                minimum=0,
                                maximum=255,
                                value=93,
                                step=1,
                                label="Threshold",
                                info="Higher values detect lighter areas as notes.",
                            )
                        # Right side - binary mask visualization
                        with gr.Column(scale=1):
                            binary_output = gr.Image(label="Binary Mask", height=200)

                # 2. Note Detection parameters with immediate visualization
                with gr.Group():
                    gr.Markdown("### 2. Note Detection")
                    gr.Markdown(
                        "Filter shapes in the binary image to identify those that represent notes."
                    )
                    # Parameters on the left
                    with gr.Row():
                        with gr.Column(scale=1):
                            min_area = gr.Slider(
                                0.01,
                                10.0,
                                1.0,
                                0.01,
                                label="Min Area",
                                info="Minimum area for a note (pixels)",
                            )
                            max_area = gr.Slider(
                                100,
                                5000,
                                5000,
                                100,
                                label="Max Area",
                                info="Maximum area for a note (pixels)",
                            )
                            min_aspect = gr.Slider(
                                0.1,
                                5.0,
                                0.1,
                                0.01,
                                label="Min Aspect Ratio",
                                info="Min width/height ratio for a note",
                            )
                            max_aspect = gr.Slider(
                                5.0,
                                50.0,
                                20.0,
                                0.5,
                                label="Max Aspect Ratio",
                                info="Max width/height ratio for a note",
                            )

                    # Note detection visualizations
                    with gr.Row():
                        note_vis_rgb = gr.Image(label="Notes on Original", height=200)
                        note_vis_bin = gr.Image(label="Notes on Binary", height=200)

                # 3. Staff Line parameters
                with gr.Group():
                    gr.Markdown("### 3. Staff Line Fitting")
                    gr.Markdown(
                        "Create musical staff lines and align the detected notes to them."
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            method = gr.Radio(
                                ["Original", "Average", "Adjustable"],
                                value="Original",
                                label="Line Fitting Method",
                                info="How to adjust note heights",
                            )
                            num_lines = gr.Slider(
                                2,
                                50,
                                10,
                                1,
                                label="Number of Lines",
                                info="More lines = more notes",
                            )
                            height_factor = gr.Slider(
                                0.0,
                                1.0,
                                0.5,
                                0.01,
                                label="Height Factor",
                                info="For Adjustable method only",
                            )

                    # Staff visualization
                    with gr.Row():
                        staff_viz = gr.Image(label="Staff Lines", height=200)
                        quant_viz = gr.Image(label="Quantized Notes", height=200)

                # 4-5. MIDI Generation and Transformations
                with gr.Group():
                    gr.Markdown("### 4. MIDI Generation & Transformations")

                    with gr.Row():
                        # Basic MIDI parameters
                        with gr.Column(scale=1):
                            gr.Markdown("#### Basic MIDI Settings")
                            base_midi = gr.Slider(
                                21,
                                108,
                                60,
                                1,
                                label="Base MIDI Note",
                                info="MIDI note for lowest staff line",
                            )
                            tempo_bpm = gr.Slider(
                                30,
                                240,
                                120,
                                1,
                                label="Tempo (BPM)",
                                info="Playback speed",
                            )

                        # Musical transformations
                        with gr.Column(scale=1):
                            gr.Markdown("#### Musical Transformations")
                            fit_to_scale = gr.Checkbox(
                                label="Fit to scale",
                                value=False,
                                info="Snap notes into a musical scale",
                            )
                            with gr.Row():
                                root_note = gr.Dropdown(
                                    choices=get_available_key_signatures(),
                                    label="Root note",
                                    value="C",
                                )
                                scale_type = gr.Dropdown(
                                    choices=get_available_scale_names(),
                                    label="Scale",
                                    value="Major",
                                )

                            quantize = gr.Checkbox(
                                label="Quantize rhythm",
                                value=False,
                                info="Snap notes to a rhythmic grid",
                            )
                            note_value = gr.Dropdown(
                                choices=list(NOTE_VALUE_TO_GRID.keys()),
                                label="Grid resolution",
                                value="Quarter",
                            )

                # Metrics and piano roll
                with gr.Group():
                    gr.Markdown("### Output")

                    # Piano roll visualization
                    piano_roll = gr.Image(label="Piano Roll Visualization", height=300)

                    # Metrics in a compact row
                    with gr.Row():
                        with gr.Column(scale=1):
                            accuracy_display = gr.Textbox(label="Fit Accuracy")
                        with gr.Column(scale=1):
                            variation_display = gr.Textbox(label="Pitch Variation")
                        with gr.Column(scale=1):
                            notes_count = gr.Textbox(label="Notes Detected")
                        with gr.Column(scale=1):
                            base_note_display = gr.Textbox(label="Base Note")

                    # MIDI playback and download
                    with gr.Row():
                        with gr.Column(scale=2):
                            midi_audio = gr.Audio(
                                label="MIDI Playback", type="filepath"
                            )
                        with gr.Column(scale=1):
                            midi_download = gr.File(
                                label="Download MIDI", type="filepath"
                            )

        # Connect all parameters to update the entire pipeline
        input_controls = [
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
            fit_to_scale,
            root_note,
            scale_type,
            quantize,
            note_value,
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

        # Create an initial state when the app loads
        interface.load(
            fn=update_pipeline_ui,
            inputs=input_controls,
            outputs=output_components,
        )

        # Connect all inputs to the update function for when parameters change
        for control in input_controls:
            control.change(
                fn=update_pipeline_ui, inputs=input_controls, outputs=output_components
            )

    return interface


if __name__ == "__main__":
    demo = create_gradio_interface()
    # Enable hot reloading for development
    demo.launch(
        share=False,  # Set to True if you want a public link
        debug=True,  # Show more error information
        show_error=True,  # Display Python errors in the UI
    )
