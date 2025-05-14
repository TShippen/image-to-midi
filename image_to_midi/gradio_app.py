# image_to_midi/gradio_app.py

import logging
import gradio as gr

from image_to_midi.app_state import (
    load_fixed_image,
    get_image_id,
)
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

# 1) Load + register the default image one time
fixed_image_array = load_fixed_image("img/paint_splatter.png")
initial_image_id = (
    get_image_id(fixed_image_array) if fixed_image_array is not None else None
)

# 2) Descriptions for the staff-fitting methods
method_descriptions = {
    "Original": "Uses original note heights as detected - maintains exact proportions.",
    "Average": "Normalizes all notes to the same height - reduces variance.",
    "Adjustable": "Custom scaling of note heights - balance consistency vs. original proportions.",
}


# 3) Helpers
def method_changed(method_value):
    """Show height_factor only if 'Adjustable'."""
    return gr.update(visible=(method_value == "Adjustable"))


def midi_note_label(x):
    """Convert a MIDI number to e.g. 'C4 (MIDI: 60)'."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note = names[x % 12] + str((x // 12) - 1)
    return f"{note} (MIDI: {x})"


def initialize_app(image_id):
    """Called on load: set every view to the defaults."""
    # 1. Binary mask
    orig, binary = update_binary_view(image_id, 93)
    # 2. Note detection
    notes_rgb, notes_bin, note_count = update_detection_view(
        image_id, 93, 1.0, 5000.0, 0.1, 20.0
    )
    # 3. Staff fitting
    staff_img, quant_img, acc, var = update_staff_view(
        image_id, 93, 1.0, 5000.0, 0.1, 20.0, "Original", 10, 0.5
    )
    # 4. MIDI generation
    p_roll, base_note, audio_path, midi_path, audio_dl_path = update_midi_view(
        image_id,
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


def create_gradio_interface():
    with gr.Blocks(title="Image to MIDI Converter") as interface:
        gr.Markdown("# 🎵 Image to MIDI Converter")
        gr.Markdown(
            "Convert a paint-splatter image to music! "
            "Tweak parameters and watch each step update in real time."
        )

        # ⚙️ This State holds our current image ID across all callbacks
        image_state = gr.State(initial_image_id)

        with gr.Row():
            with gr.Column(scale=1):
                # Display the fixed image
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
                        0,
                        255,
                        value=93,
                        step=1,
                        label="Threshold",
                        info="Higher: detect lighter spots as notes.",
                    )
                    with gr.Row():
                        original_view = gr.Image(label="Original Image", height=200)
                        binary_output = gr.Image(label="Binary Mask", height=200)

                # 2. Note Detection
                with gr.Group():
                    gr.Markdown("### 2. Note Detection")
                    with gr.Row():
                        min_area = gr.Slider(
                            0.01, 10.0, value=1.0, step=0.01, label="Min Area"
                        )
                        max_area = gr.Slider(
                            100, 5000, value=5000, step=100, label="Max Area"
                        )
                    with gr.Row():
                        min_aspect = gr.Slider(
                            0.1, 5.0, value=0.1, step=0.01, label="Min Aspect Ratio"
                        )
                        max_aspect = gr.Slider(
                            5.0, 50.0, value=20.0, step=0.5, label="Max Aspect Ratio"
                        )
                    note_count_display = gr.Textbox(
                        label="Detected Notes", value="No notes detected yet"
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
                    )
                    method_description = gr.Markdown()
                    num_lines = gr.Slider(
                        2, 50, value=10, step=1, label="Number of Staff Lines"
                    )
                    height_factor_container = gr.Group(visible=False)
                    with height_factor_container:
                        height_factor = gr.Slider(
                            0.0, 1.0, value=0.5, step=0.01, label="Height Factor"
                        )
                    staff_accuracy = gr.Textbox(
                        label="Fit Accuracy", value="Waiting for processing..."
                    )
                    pitch_variation = gr.Textbox(
                        label="Pitch Variation", value="Waiting for processing..."
                    )
                    with gr.Row():
                        staff_viz = gr.Image(label="Staff Lines", height=200)
                        quant_viz = gr.Image(label="Quantized Notes", height=200)

                # 4. MIDI Generation & Transformations
                with gr.Group():
                    gr.Markdown("### 4. MIDI Generation & Transformations")
                    base_midi = gr.Slider(
                        21, 108, value=60, step=1, label="Base MIDI Note"
                    )
                    base_note_display = gr.Textbox(
                        label="Base Note", value=midi_note_label(60)
                    )
                    tempo_bpm = gr.Slider(
                        30, 240, value=120, step=1, label="Tempo (BPM)"
                    )
                    fit_to_scale = gr.Checkbox(label="Fit to scale", value=False)
                    root_note = gr.Dropdown(
                        choices=get_available_key_signatures(),
                        value="C",
                        label="Root note",
                    )
                    scale_type = gr.Dropdown(
                        choices=get_available_scale_names(),
                        value="Major",
                        label="Scale",
                    )
                    quantize = gr.Checkbox(label="Quantize rhythm", value=False)
                    note_value = gr.Dropdown(
                        choices=list(NOTE_VALUE_TO_GRID.keys()),
                        value="Quarter",
                        label="Grid resolution",
                    )
                    piano_roll = gr.Image(label="Piano Roll Visualization", height=500)
                    with gr.Row():
                        audio_player = gr.Audio(
                            label="Listen to the music", type="filepath"
                        )
                        midi_download = gr.File(
                            label="Download MIDI File", type="filepath"
                        )
                        audio_download = gr.File(
                            label="Download Audio File", type="filepath"
                        )
                    gr.Markdown(
                        "_Note: Audio playback quality depends on your browser. "
                        "For best results, download and open in a dedicated player._"
                    )

        # All inputs and outputs, reused below
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

        # —— INITIALIZATION on load ——
        interface.load(
            fn=initialize_app,
            inputs=[image_state],
            outputs=output_components,
        )

        # —— PARAMETER CALLBACKS ——
        method.change(
            fn=lambda x: method_descriptions[x],
            inputs=[method],
            outputs=[method_description],
        )
        method.change(
            fn=method_changed, inputs=[method], outputs=[height_factor_container]
        )
        base_midi.change(
            fn=midi_note_label, inputs=[base_midi], outputs=[base_note_display]
        )

        # 1. Binary mask only needs threshold + image
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
                inputs=[image_state]
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

    return interface


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        share=False,
        debug=True,
        show_error=True,
        server_port=7860,
    )
