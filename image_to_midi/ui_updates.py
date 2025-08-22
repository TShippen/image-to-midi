"""UI update functions for the Gradio interface.

This module provides cached update functions that interface between the
Gradio UI components and the image-to-MIDI processing pipeline. Each function
corresponds to a specific stage of processing and returns the appropriate
visualizations and data for display in the web interface.

All functions use LRU caching to improve responsiveness when users adjust
parameters, building upon the cached pipeline functions for maximum efficiency.
"""

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
    create_piano_roll_visualization,
)

from image_to_midi.midi_utils import midi_to_audio
from image_to_midi.music_transformations import NOTE_VALUE_TO_GRID

# Global constant
from image_to_midi.app_state import get_image_by_id


@lru_cache(maxsize=32)
def update_binary_view(image_id: str, threshold: int) -> tuple:
    """Update binary processing visualization for the UI.

    Processes an image to create a binary mask and returns both the
    original image and the binary visualization for display in the
    Gradio interface.

    Args:
        image_id: Unique identifier for the registered image.
        threshold: Binarization threshold value (0-255).

    Returns:
        Tuple of (original_image, binary_visualization) where both
        are NumPy arrays suitable for Gradio image display.
    """
    image = get_image_by_id(image_id)
    binary_result = cached_binary_processing(image_id, threshold)

    # Create visualization
    binary_viz = create_binary_visualization(binary_result.binary_mask)

    return image, binary_viz


@lru_cache(maxsize=32)
def update_detection_view(
    image_id: str,
    threshold: int,
    min_area: float,
    max_area: float,
    min_aspect: float,
    max_aspect: float,
) -> tuple:
    """Update note detection visualization for the UI.

    Processes an image through binary processing and note detection,
    then creates visualizations showing detected note bounding boxes
    overlaid on both the original and binary images.

    Args:
        image_id: Unique identifier for the registered image.
        threshold: Binarization threshold from binary processing.
        min_area: Minimum acceptable contour area in pixels.
        max_area: Maximum acceptable contour area in pixels.
        min_aspect: Minimum acceptable width/height ratio.
        max_aspect: Maximum acceptable width/height ratio.

    Returns:
        Tuple of (note_rgb_overlay, note_binary_overlay, note_count_text)
        where overlays are NumPy arrays and count_text is a formatted string.
    """
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
    image_id: str,
    threshold: int,
    min_area: float,
    max_area: float,
    min_aspect: float,
    max_aspect: float,
    method: str,
    num_lines: int,
    height_factor: float,
) -> tuple:
    """Update staff line and quantization visualization for the UI.

    Processes an image through all stages up to staff creation,
    generating visualizations that show staff lines with both
    original and quantized note positions, plus quality metrics.

    Args:
        image_id: Unique identifier for the registered image.
        threshold: Binarization threshold from binary processing.
        min_area: Minimum acceptable contour area from detection.
        max_area: Maximum acceptable contour area from detection.
        min_aspect: Minimum acceptable aspect ratio from detection.
        max_aspect: Maximum acceptable aspect ratio from detection.
        method: Staff fitting method ("Original", "Average", or "Adjustable").
        num_lines: Number of staff lines to generate.
        height_factor: Height adjustment factor for "Adjustable" method.

    Returns:
        Tuple of (staff_original_viz, staff_quantized_viz, accuracy_text, variation_text)
        where visualizations are NumPy arrays and metrics are formatted strings.
    """
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
    image_id: str,
    threshold: int,
    min_area: float,
    max_area: float,
    min_aspect: float,
    max_aspect: float,
    method: str,
    num_lines: int,
    height_factor: float,
    base_midi: int,
    tempo_bpm: int,
    fit_to_scale: bool,
    root_note: str,
    scale_type: str,
    quantize: bool,
    note_value: str,
) -> tuple:
    """Update MIDI generation visualization and create downloadable files.

    Processes an image through the complete pipeline to generate MIDI output,
    creates a piano roll visualization, synthesizes audio, and prepares
    downloadable files for the user.

    Args:
        image_id: Unique identifier for the registered image.
        threshold: Binarization threshold from binary processing.
        min_area: Minimum acceptable contour area from detection.
        max_area: Maximum acceptable contour area from detection.
        min_aspect: Minimum acceptable aspect ratio from detection.
        max_aspect: Maximum acceptable aspect ratio from detection.
        method: Staff fitting method from staff creation.
        num_lines: Number of staff lines from staff creation.
        height_factor: Height adjustment factor from staff creation.
        base_midi: Base MIDI note number for pitch mapping.
        tempo_bpm: Tempo in beats per minute.
        fit_to_scale: Whether to apply scale mapping transformation.
        root_note: Root key for scale mapping (if enabled).
        scale_type: Scale type for mapping (if enabled).
        quantize: Whether to apply rhythm quantization.
        note_value: Note value for rhythm quantization grid (if enabled).

    Returns:
        Tuple of (piano_roll_fig, base_note_display, audio_path, midi_path, audio_path_duplicate)
        where piano_roll_fig is a matplotlib Figure, display text is a string,
        and paths are file paths for download/playback.
    """
    # Convert note value to grid size and quantization strength
    grid_size = NOTE_VALUE_TO_GRID[note_value]
    quantize_strength = 1.0 if quantize else 0.0

    # Get cached MIDI generation result
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

    # Create piano roll visualization from MIDI events
    piano_roll = None
    if midi_result.events:
        piano_roll = create_piano_roll_visualization(midi_result.events)

    # Write MIDI bytes to temporary file and synthesize to audio
    midi_download_path = None
    audio_play_path = None

    if midi_result.midi_bytes:
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp.write(midi_result.midi_bytes)
            midi_download_path = tmp.name

        audio_play_path = midi_to_audio(midi_download_path)
        if audio_play_path is None:
            audio_play_path = midi_download_path

    # Create base note display string for UI
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (base_midi // 12) - 1
    note_name = note_names[base_midi % 12]
    base_note_display = f"{note_name}{octave} (MIDI: {base_midi})"

    # Return visualization and file paths for Gradio interface
    return (
        piano_roll,
        base_note_display,
        audio_play_path,
        midi_download_path,
        audio_play_path,
    )
