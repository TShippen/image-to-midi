"""Gradio web interface for the image-to-MIDI converter.

This module creates and configures the main Gradio web application interface
for the image-to-MIDI conversion pipeline. It provides an interactive web UI
where users can adjust processing parameters and see real-time updates of
visualizations and outputs from each pipeline stage.

The interface is organized into sections corresponding to each processing stage:
- Binary image processing with threshold control
- Note detection with area and aspect ratio filters  
- Staff line generation with various fitting methods
- MIDI generation with musical transformation options
"""

import logging
from uuid import uuid4

import gradio as gr

from image_to_midi.app_state import load_fixed_image, get_image_id
from image_to_midi.ui_updates import (
    update_binary_view,
    update_detection_view,
    update_staff_view,
    update_midi_view,
)
from image_to_midi.music_transformations import (
    get_available_key_signatures,
    get_available_scale_names,
    NOTE_VALUE_TO_GRID,
)

logger = logging.getLogger(__name__)

# 1) Load + register the default image once
fixed_image_array = load_fixed_image("img/paint_splatter.png")
initial_image_id = (
    get_image_id(fixed_image_array) if fixed_image_array is not None else None
)

# 2) Descriptions for the staff-fitting methods
method_descriptions = {
    "Original": "Uses original note heights as detected – maintains exact proportions of detected shapes.",
    "Average": "Normalizes all notes to the same height – reduces pitch variance but may lose some distinctions.",
    "Adjustable": "Allows custom scaling of note heights – balance consistency vs. original proportions.",
}


# 3) Helpers
def method_changed(method_value: str) -> gr.update:
    """Control visibility of Height Factor slider based on staff method selection.

    Args:
        method_value: Selected staff fitting method string.

    Returns:
        Gradio update object controlling slider visibility.
    """
    return gr.update(visible=(method_value == "Adjustable"))


def midi_note_label(x: int) -> str:
    """Convert a MIDI note number to human-readable format.

    Args:
        x: MIDI note number (0-127).

    Returns:
        Formatted string like 'C4 (MIDI: 60)' showing note name, octave, and number.
    """
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note = names[x % 12] + str((x // 12) - 1)
    return f"{note} (MIDI: {x})"


def initialize_app(image_id: str, session_id: str) -> list:
    """Initialize all pipeline visualizations with default parameter values.

    Processes the registered image through all pipeline stages using default
    parameters to generate initial visualizations for the Gradio interface.

    Args:
        image_id: Unique identifier for the registered image.
        session_id: Unique session identifier for file management isolation.

    Returns:
        List of visualization outputs for populating the Gradio interface,
        including images, plots, text displays, and file paths.
    """
    orig, binary = update_binary_view(image_id, 93)
    notes_rgb, notes_bin, note_count = update_detection_view(
        image_id, 93, 1.0, 5000.0, 0.1, 20.0
    )
    staff_img, quant_img, acc, var = update_staff_view(
        image_id, 93, 1.0, 5000.0, 0.1, 20.0, "Original", 10, 0.5
    )
    p_roll, base_note, audio_path, midi_path, audio_dl_path = update_midi_view(
        image_id,
        session_id,
        93,
        1.0,
        5000.0,
        0.1,
        20.0,
        "Original",
        10,
        0.5,
        60,
        120,
        False,
        "C",
        "Major",
        False,
        "Quarter",
    )
    return [
        orig,
        binary,
        notes_rgb,
        notes_bin,
        note_count,
        staff_img,
        quant_img,
        acc,
        var,
        p_roll,
        base_note,
        audio_path,
        midi_path,
        audio_dl_path,
    ]


def create_gradio_interface() -> gr.Blocks:
    """Create and configure the main Gradio web interface.

    Builds the complete web application interface with all UI components,
    event handlers, and interactive controls for the image-to-MIDI conversion
    pipeline. The interface is organized into tabbed sections corresponding
    to each processing stage with real-time parameter adjustment and visualization.

    Returns:
        Configured Gradio Blocks interface ready for launching.
    """
    # Configure automatic cache cleanup: check every 30 minutes, delete files older than 1 hour
    with gr.Blocks(
        title="Image to MIDI Converter",
        delete_cache=(1800, 3600)
    ) as interface:
        gr.Markdown("# 🎵 Image to MIDI Converter")
        gr.Markdown(
            "Convert a paint-splatter image to music! "
            "Adjust parameters and watch each step update in real time."
        )

        # Holds the current image ID across callbacks
        image_state = gr.State(initial_image_id)

        # Unique session ID for per-session file management
        session_state = gr.State(str(uuid4()))

        with gr.Row():
            with gr.Column(scale=1):
                gr.Image(
                    value=fixed_image_array,
                    label="Source Image",
                    interactive=False,
                    height=200,
                )

                # 1. Image Processing
                with gr.Group():
                    gr.Markdown("### 1. Image Processing")
                    threshold = gr.Slider(
                        minimum=0,
                        maximum=255,
                        value=93,
                        step=1,
                        label="Threshold",
                        info="Higher values detect lighter spots as notes.",
                    )
                    with gr.Row():
                        original_view = gr.Image(label="Original Image", height=200)
                        binary_output = gr.Image(label="Binary Mask", height=200)

                # 2. Note Detection
                with gr.Group():
                    gr.Markdown("### 2. Note Detection")
                    with gr.Row():
                        min_area = gr.Slider(
                            0.01,
                            10.0,
                            value=1.0,
                            step=0.01,
                            label="Min Area",
                            info="Minimum area (pixels) for a detected note.",
                        )
                        max_area = gr.Slider(
                            100,
                            5000,
                            value=5000,
                            step=100,
                            label="Max Area",
                            info="Maximum area (pixels) for a detected note.",
                        )
                    with gr.Row():
                        min_aspect = gr.Slider(
                            0.1,
                            5.0,
                            value=0.1,
                            step=0.01,
                            label="Min Aspect Ratio",
                            info="Minimum width/height ratio for a detected note.",
                        )
                        max_aspect = gr.Slider(
                            5.0,
                            50.0,
                            value=20.0,
                            step=0.5,
                            label="Max Aspect Ratio",
                            info="Maximum width/height ratio for a detected note.",
                        )
                    note_count_display = gr.Textbox(
                        label="Detected Notes",
                        value="No notes detected yet",
                    )
                    with gr.Row():
                        note_vis_rgb = gr.Image(label="Notes on Original", height=200)
                        note_vis_bin = gr.Image(label="Notes on Binary", height=200)

                # 3. Staff Line Fitting
                with gr.Group():
                    gr.Markdown("### 3. Staff Line Fitting")
                    method = gr.Radio(
                        choices=list(method_descriptions.keys()),
                        value="Original",
                        label="Line Fitting Method",
                        info="Select how to adjust note heights when fitting staff lines.",
                    )
                    method_description = gr.Markdown(method_descriptions["Original"])
                    with gr.Row():
                        num_lines = gr.Slider(
                            2,
                            50,
                            value=10,
                            step=1,
                            label="Number of Staff Lines",
                            info="Number of lines to fit (more lines = finer pitch resolution).",
                        )
                        height_factor_container = gr.Group(visible=False)
                        with height_factor_container:
                            height_factor = gr.Slider(
                                0.0,
                                1.0,
                                value=0.5,
                                step=0.01,
                                label="Height Factor",
                                info="Controls hybrid scaling: 0=uniform, 1=original heights.",
                            )
                    staff_accuracy = gr.Textbox(
                        label="Fit Accuracy",
                        value="Waiting for processing...",
                    )
                    pitch_variation = gr.Textbox(
                        label="Pitch Variation",
                        value="Waiting for processing...",
                    )
                    with gr.Row():
                        staff_viz = gr.Image(label="Staff Lines", height=200)
                        quant_viz = gr.Image(label="Quantized Notes", height=200)

                # 4. MIDI Generation & Transformations
                with gr.Group():
                    gr.Markdown("### 4. MIDI Generation & Transformations")
                    base_midi = gr.Slider(
                        21,
                        108,
                        value=60,
                        step=1,
                        label="Base MIDI Note",
                        info="MIDI note number for the lowest staff line.",
                    )
                    base_note_display = gr.Textbox(
                        label="Base Note",
                        value=midi_note_label(60),
                    )
                    tempo_bpm = gr.Slider(
                        30,
                        240,
                        value=120,
                        step=1,
                        label="Tempo (BPM)",
                        info="Playback speed in beats per minute.",
                    )
                    fit_to_scale = gr.Checkbox(
                        label="Fit to Scale",
                        value=False,
                        info="Snap detected notes to the chosen musical scale.",
                    )
                    root_note = gr.Dropdown(
                        choices=get_available_key_signatures(),
                        value="C",
                        label="Root Note",
                        info="Root note of the musical scale.",
                    )
                    scale_type = gr.Dropdown(
                        choices=get_available_scale_names(),
                        value="Major",
                        label="Scale Type",
                        info="Scale type (e.g., Major, Minor) for fitting notes.",
                    )
                    quantize = gr.Checkbox(
                        label="Quantize Rhythm",
                        value=False,
                        info="Snap note start times to a rhythmic grid.",
                    )
                    note_value = gr.Dropdown(
                        choices=list(NOTE_VALUE_TO_GRID.keys()),
                        value="Quarter",
                        label="Grid Resolution",
                        info="Rhythmic resolution for quantization.",
                    )
                    piano_roll = gr.Plot(
                        label="Piano Roll Visualization", elem_id="piano-roll-plot"
                    )
                    with gr.Row():
                        audio_player = gr.Audio(
                            label="Listen to the Music",
                            type="filepath",
                            format="wav",
                            autoplay=False
                        )
                        midi_download = gr.File(
                            label="Download MIDI File",
                            type="filepath",
                        )
                        audio_download = gr.File(
                            label="Download Audio File",
                            type="filepath",
                        )
                    gr.Markdown(
                        "_Note: Audio playback quality depends on your browser. "
                        "For best results, download and open in a dedicated player._"
                    )

        # Group input lists for callbacks
        note_detection_params = [threshold, min_area, max_area, min_aspect, max_aspect]
        staff_params = [method, num_lines, height_factor]
        midi_params = [
            base_midi,
            tempo_bpm,
            fit_to_scale,
            root_note,
            scale_type,
            quantize,
            note_value,
        ]
        output_components = [
            original_view,
            binary_output,
            note_vis_rgb,
            note_vis_bin,
            note_count_display,
            staff_viz,
            quant_viz,
            staff_accuracy,
            pitch_variation,
            piano_roll,
            base_note_display,
            audio_player,
            midi_download,
            audio_download,
        ]

        # INITIALIZE on page load
        interface.load(
            fn=initialize_app,
            inputs=[image_state, session_state],
            outputs=output_components,
        )

        # STAFF METHOD description + visibility
        method.change(
            fn=lambda x: method_descriptions[x],
            inputs=[method],
            outputs=[method_description],
        )
        method.change(
            fn=method_changed,
            inputs=[method],
            outputs=[height_factor_container],
        )

        # BASE NOTE display
        base_midi.change(
            fn=midi_note_label,
            inputs=[base_midi],
            outputs=[base_note_display],
        )

        # 1. Binary mask updates
        threshold.change(
            fn=update_binary_view,
            inputs=[image_state, threshold],
            outputs=[original_view, binary_output],
        )

        # 2. Note detection updates
        for p in note_detection_params:
            p.change(
                fn=update_detection_view,
                inputs=[image_state] + note_detection_params,
                outputs=[note_vis_rgb, note_vis_bin, note_count_display],
            )

        # 3. Staff fitting updates
        for p in note_detection_params + staff_params:
            p.change(
                fn=update_staff_view,
                inputs=[image_state] + note_detection_params + staff_params,
                outputs=[staff_viz, quant_viz, staff_accuracy, pitch_variation],
            )

        # 4. MIDI updates
        for p in note_detection_params + staff_params + midi_params:
            p.change(
                fn=update_midi_view,
                inputs=[image_state, session_state]
                + note_detection_params
                + staff_params
                + midi_params,
                outputs=[
                    piano_roll,
                    base_note_display,
                    audio_player,
                    midi_download,
                    audio_download,
                ],
            )

    # Add cleanup handler for when users disconnect
    def cleanup_session_handler(session_id: str) -> None:
        """Clean up session files when user disconnects.

        Args:
            session_id: Unique session identifier to clean up.
        """
        try:
            from image_to_midi.ui_updates import cleanup_session
            cleanup_session(session_id)
        except Exception as e:
            # Log error but don't disrupt user experience
            logger.warning(f"Cleanup failed for session {session_id}: {e}")

    # Register unload event for immediate cleanup (no 60-minute delay)
    interface.unload(cleanup_session_handler, inputs=[session_state])
    
    return interface


if __name__ == "__main__":
    import logging
    import asyncio
    import sys

    # Reduce logging verbosity for asyncio to suppress connection noise
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Windows-specific: Use SelectorEventLoop to avoid ProactorEventLoop issues
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    demo = create_gradio_interface()
    demo.launch(
        share=False,
        debug=True,
        show_error=True,
        server_port=7860,
    )
