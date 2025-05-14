# ui_updates.py
"""UI update functions for Gradio."""

import tempfile
import cv2
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
    create_piano_roll_visualization,
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
    # Convert parameters
    grid_size = NOTE_VALUE_TO_GRID[note_value]
    quantize_strength = 1.0 if quantize else 0.0

    # Get cached MIDI result
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

    # Create piano roll visualization
    piano_roll = (
        create_piano_roll_visualization(midi_result.events)
        if midi_result.events
        else None
    )

    # Enhance piano roll visualization if needed
    if piano_roll is not None:
        try:
            # Adjust aspect ratio for better visibility
            h, w = piano_roll.shape[:2]
            if w > h * 2:  # If width is more than twice the height
                new_h = min(h * 2, 800)  # Double height, cap at 800px
                new_w = w * 3 // 4  # Reduce width by 25%
                piano_roll = cv2.resize(
                    piano_roll, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
        except Exception as e:
            print(f"Could not resize piano roll: {str(e)}")

    # Create files for playback and download
    midi_download_path = None
    audio_play_path = None

    if midi_result.midi_bytes:
        # Create a temporary file for MIDI download
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp.write(midi_result.midi_bytes)
            midi_download_path = tmp.name

        # Try to convert to audio for playback
        audio_play_path = midi_to_audio(midi_download_path)
        if audio_play_path is None:
            # If conversion fails, use the MIDI file as fallback
            audio_play_path = midi_download_path

    # Get note name
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (base_midi // 12) - 1
    note_name = note_names[base_midi % 12]
    base_note_display = f"{note_name}{octave} (MIDI: {base_midi})"

    return (
        piano_roll,
        base_note_display,
        audio_play_path,
        midi_download_path,
        audio_play_path,
    )
