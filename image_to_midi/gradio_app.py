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
        return [None] * 15  # Updated for additional outputs

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

    # Get base note name
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (base_midi // 12) - 1
    note_name = note_names[base_midi % 12]
    base_note_display = f"{note_name}{octave} (MIDI: {base_midi})"

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
    midi_play_path = None
    if midi_result.midi_bytes:
        # Create a temporary file with .mid extension for download
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp.write(midi_result.midi_bytes)
            midi_download_path = tmp.name

        # Create another temp file for playback to ensure it's accessible
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp.write(midi_result.midi_bytes)
            midi_play_path = tmp.name

    # Notes detected count for display in the Note Detection section
    note_count_display = f"{len(detection_result.note_boxes)} notes detected"

    # Format outputs for Gradio interface
    return (
        # Original image for comparison in section 1
        image,
        # Section 1 - Binary mask
        vis_set.binary_mask,
        # Section 2 - Note detection
        vis_set.note_detection,
        vis_set.note_detection_binary,
        note_count_display,  # Notes count for section 2
        # Section 3 - Staff lines
        vis_set.staff_lines,
        vis_set.quantized_notes,
        f"{staff_result.fit_accuracy:.2f}%",  # Fit accuracy for section 3
        f"{staff_result.pitch_variation:.2f}",  # Pitch variation for section 3
        # Section 4 - MIDI output
        vis_set.piano_roll,
        # Output section
        base_note_display,  # Base note for display
        note_count_display,  # Notes count for output section
        f"{staff_result.fit_accuracy:.2f}%",  # Fit accuracy for output section
        midi_play_path,  # Use separate path for playback
        midi_download_path,  # Use the temporary path for the download link
    )


# Helper function to handle method description and conditional UI
def method_changed(method_value):
    if method_value == "Adjustable":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def create_gradio_interface():
    """Create and configure the Gradio interface for the Image-to-MIDI app."""
    with gr.Blocks(title="Image to MIDI Converter") as interface:
        gr.Markdown("# 🎵 Image to MIDI Converter")
        gr.Markdown(
            "This app converts a paint splatter image to music! "
            "Adjust the parameters to see how they affect the entire process."
        )

        # Define method descriptions
        method_descriptions = {
            "Original": "Uses original note heights as detected - maintains the exact proportions of the detected shapes.",
            "Average": "Normalizes all notes to the same height - reduces variance but may lose some intended pitch distinctions.",
            "Adjustable": "Allows custom scaling of note heights - find a balance between original proportions and consistency.",
        }

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
                    gr.Markdown(
                        "Control how the image is converted to a binary mask that identifies potential notes."
                    )

                    # Controls
                    threshold = gr.Slider(
                        minimum=0,
                        maximum=255,
                        value=93,
                        step=1,
                        label="Threshold",
                        info="Higher values detect lighter areas as notes.",
                    )

                    # Side-by-side comparison of original and binary
                    with gr.Row():
                        original_view = gr.Image(label="Original Image", height=200)
                        binary_output = gr.Image(label="Binary Mask", height=200)

                # 2. Note Detection parameters with immediate visualization
                with gr.Group():
                    gr.Markdown("### 2. Note Detection")
                    gr.Markdown(
                        "Filter shapes in the binary image to identify those that represent notes."
                    )

                    # Parameters
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

                        with gr.Column(scale=1):
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

                    # Note count display
                    note_count_display = gr.Textbox(
                        label="Detected Notes", value="No notes detected yet"
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

                    # Method selection with descriptions
                    method = gr.Radio(
                        ["Original", "Average", "Adjustable"],
                        value="Original",
                        label="Line Fitting Method",
                        info="Select how to adjust note heights for staff fitting",
                    )

                    # Method description display
                    method_description = gr.Markdown()

                    # Update the description when method changes
                    method.change(
                        fn=lambda x: method_descriptions[x],
                        inputs=[method],
                        outputs=[method_description],
                    )

                    with gr.Row():
                        with gr.Column(scale=1):
                            num_lines = gr.Slider(
                                2,
                                50,
                                10,
                                1,
                                label="Number of Staff Lines",
                                info="More lines = more notes",
                            )

                            # Height factor (conditionally visible)
                            height_factor_container = gr.Group()
                            with height_factor_container:
                                height_factor = gr.Slider(
                                    0.0,
                                    1.0,
                                    0.5,
                                    0.01,
                                    label="Height Factor",
                                    info="Controls note height adjustment (0=uniform, 1=original)",
                                )

                            # Make height factor visible only when Adjustable is selected
                            method.change(
                                fn=method_changed,
                                inputs=[method],
                                outputs=[height_factor_container],
                            )

                        # Staff metrics
                        with gr.Column(scale=1):
                            staff_accuracy = gr.Textbox(
                                label="Fit Accuracy", value="Waiting for processing..."
                            )
                            pitch_variation = gr.Textbox(
                                label="Pitch Variation",
                                value="Waiting for processing...",
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
                            base_note_display = gr.Textbox(
                                label="Base Note", value="C4 (MIDI: 60)"
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

                # Output section
                with gr.Group():
                    gr.Markdown("### MIDI Output & Piano Roll")

                    # Piano roll visualization with increased size
                    piano_roll = gr.Image(label="Piano Roll Visualization", height=400)

                    # MIDI playback and download
                    with gr.Row():
                        with gr.Column(scale=2):
                            midi_audio = gr.Audio(
                                label="MIDI Playback",
                                type="filepath",
                                elem_id="midi_player",
                            )
                        with gr.Column(scale=1):
                            midi_download = gr.File(
                                label="Download MIDI",
                                type="filepath",
                                elem_id="midi_download",
                            )

                    # Add a note about MIDI playback
                    gr.Markdown(
                        "_Note: Some browsers may have limited MIDI playback support. "
                        "For best results, download the MIDI file and open it in a music player._"
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
            # Section 1 outputs
            original_view,
            binary_output,
            # Section 2 outputs
            note_vis_rgb,
            note_vis_bin,
            note_count_display,
            # Section 3 outputs
            staff_viz,
            quant_viz,
            staff_accuracy,
            pitch_variation,
            # Section 4 and output section
            piano_roll,
            base_note_display,
            # Additional outputs for other sections
            note_count_display,
            staff_accuracy,
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

        # Update Base Note display when Base MIDI changes
        base_midi.change(
            fn=lambda x: f"{['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][x % 12]}{(x // 12) - 1} (MIDI: {x})",
            inputs=[base_midi],
            outputs=[base_note_display],
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
