# ui_updates.py
"""UI update functions for Gradio."""

import tempfile
from functools import lru_cache

from image_to_midi.cache import (
    cached_binary_processing,
    cached_note_detection,
    cached_staff_creation,
    cached_midi_generation,
)

from image_to_midi.visualization import (
    create_binary_visualization,
    create_detection_visualizations,
    create_staff_result_visualizations,
    create_piano_roll_from_events,
    create_piano_roll_from_boxes,
)

from image_to_midi.midi_utils import midi_to_audio
from image_to_midi.music_transformations import NOTE_VALUE_TO_GRID

# Global constant
from image_to_midi.app_state import get_image_by_id


@lru_cache(maxsize=32)
def update_binary_view(image_id, threshold):
    """Update binary visualization only."""
    image = get_image_by_id(image_id)
    binary_result = cached_binary_processing(image_id, threshold)

    # Create visualization
    binary_viz = create_binary_visualization(binary_result.binary_mask)

    return image, binary_viz


@lru_cache(maxsize=32)
def update_detection_view(
    image_id, threshold, min_area, max_area, min_aspect, max_aspect
):
    """Update detection visualization only."""
    image = get_image_by_id(image_id)
    binary_result = cached_binary_processing(image_id, threshold)
    detection_result = cached_note_detection(
        image_id, threshold, min_area, max_area, min_aspect, max_aspect
    )

    # Create visualizations
    note_rgb, note_bin = create_detection_visualizations(
        image, binary_result, detection_result
    )

    note_count = f"{len(detection_result.note_boxes)} notes detected"
    return note_rgb, note_bin, note_count


@lru_cache(maxsize=32)
def update_staff_view(
    image_id,
    threshold,
    min_area,
    max_area,
    min_aspect,
    max_aspect,
    method,
    num_lines,
    height_factor,
):
    """Update staff visualization only."""
    image = get_image_by_id(image_id)
    staff_result = cached_staff_creation(
        image_id,
        threshold,
        min_area,
        max_area,
        min_aspect,
        max_aspect,
        method,
        num_lines,
        height_factor,
    )

    # Create visualizations
    staff_rgb, staff_quant = create_staff_result_visualizations(
        image.shape[:2], staff_result
    )

    accuracy = f"{staff_result.fit_accuracy:.2f}%"
    variation = f"{staff_result.pitch_variation:.2f}"
    return staff_rgb, staff_quant, accuracy, variation


@lru_cache(maxsize=16)
def update_midi_view(
    image_id,
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
    """Update MIDI visualization and outputs."""
    # 1) Convert note‐value → grid_size, quantize strength
    grid_size = NOTE_VALUE_TO_GRID[note_value]
    quantize_strength = 1.0 if quantize else 0.0

    # 2) Pull cached MidiResult
    midi_result = cached_midi_generation(
        image_id,
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
        scale_type.lower(),
        quantize,
        grid_size,
        quantize_strength,
    )

    # 3) Build piano_roll from events if there are any
    piano_roll = None
    if midi_result.events:
        piano_roll = create_piano_roll_from_events(
            midi_result.events, width_px=1400, dpi=180
        )
    else:
        # 4) Fallback: re‐fetch StaffResult so we can plot boxes→Multitrack
        staff_result = cached_staff_creation(
            image_id,
            threshold,
            min_area,
            max_area,
            min_aspect,
            max_aspect,
            method,
            num_lines,
            height_factor,
        )
        if (
            staff_result
            and staff_result.quantized_boxes
            and staff_result.lines is not None
            and staff_result.lines.size > 0
        ):
            piano_roll = create_piano_roll_from_boxes(
                boxes=staff_result.quantized_boxes,
                lines=staff_result.lines,
                base_midi=base_midi,  # same base MIDI
                ticks_per_px=4,  # must match build_note_events
                resolution=480,
                tempo=tempo_bpm,
                width_px=1400,
                dpi=180,
            )

    # 5) Write MIDI bytes to a temp file and (optionally) synthesize to WAV
    midi_download_path = None
    audio_play_path = None

    if midi_result.midi_bytes:
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp.write(midi_result.midi_bytes)
            midi_download_path = tmp.name

        audio_play_path = midi_to_audio(midi_download_path)
        if audio_play_path is None:
            audio_play_path = midi_download_path

    # 6) Compute base‐note display string
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (base_midi // 12) - 1
    note_name = note_names[base_midi % 12]
    base_note_display = f"{note_name}{octave} (MIDI: {base_midi})"

    # 7) **Return exactly these five values**.
    #    Make sure piano_roll is first in the tuple:
    return (
        piano_roll,
        base_note_display,
        audio_play_path,
        midi_download_path,
        audio_play_path,
    )
