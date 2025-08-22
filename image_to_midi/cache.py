"""Caching mechanisms for the image-to-MIDI pipeline.

This module provides cached versions of all pipeline processing functions
to improve performance when processing the same image with similar parameters.
Each stage uses LRU caching with appropriate cache sizes based on computational
complexity and memory requirements.

The caching strategy builds upon previous stages, so that changing only later-stage
parameters can reuse earlier cached results without recomputing the entire pipeline.
"""

from functools import lru_cache

from image_to_midi.models import (
    ImageProcessingParams,
    NoteDetectionParams,
    StaffParams,
    MidiParams,
)
from image_to_midi.pipeline import (
    process_binary_image,
    detect_notes,
    create_staff,
    generate_midi,
)

# Make caches with different sizes for different stages
BINARY_CACHE_SIZE = 32
DETECTION_CACHE_SIZE = 32
STAFF_CACHE_SIZE = 32
MIDI_CACHE_SIZE = 16  # Fewer entries needed as this is the most complex


# Stage 1: Binary Processing
@lru_cache(maxsize=BINARY_CACHE_SIZE)
def cached_binary_processing(image_id: str, threshold: int):
    """Cached version of binary image processing.

    Retrieves the image by ID and processes it to create a binary mask.
    Results are cached based on image ID and threshold parameters.

    Args:
        image_id: Unique identifier for the registered image.
        threshold: Binarization threshold value (0-255).

    Returns:
        BinaryResult object containing the processed binary mask.
    """
    # Get the image from your global storage
    from image_to_midi.app_state import get_image_by_id

    image = get_image_by_id(image_id)

    params = ImageProcessingParams(threshold=threshold)
    return process_binary_image(image, params)


# Stage 2: Note Detection
@lru_cache(maxsize=DETECTION_CACHE_SIZE)
def cached_note_detection(
    image_id: str,
    threshold: int,
    min_area: float,
    max_area: float,
    min_aspect: float,
    max_aspect: float,
):
    """Cached version of note detection processing.

    Builds upon cached binary processing results to detect note-like shapes.
    Results are cached based on all input parameters to avoid redundant
    contour analysis when only later-stage parameters change.

    Args:
        image_id: Unique identifier for the registered image.
        threshold: Binarization threshold from binary processing stage.
        min_area: Minimum acceptable contour area in pixels.
        max_area: Maximum acceptable contour area in pixels.
        min_aspect: Minimum acceptable width/height ratio.
        max_aspect: Maximum acceptable width/height ratio.

    Returns:
        DetectionResult object containing detected note bounding boxes.
    """
    # Get cached binary result
    binary_result = cached_binary_processing(image_id, threshold)

    # Create params and process
    params = NoteDetectionParams(
        min_area=min_area,
        max_area=max_area,
        min_aspect_ratio=min_aspect,
        max_aspect_ratio=max_aspect,
    )
    return detect_notes(binary_result, params)


# Stage 3: Staff Creation
@lru_cache(maxsize=STAFF_CACHE_SIZE)
def cached_staff_creation(
    image_id: str,
    threshold: int,
    min_area: float,
    max_area: float,
    min_aspect: float,
    max_aspect: float,
    method: str,
    num_lines: int,
    height_factor: float,
):
    """Cached version of staff line creation and note quantization.

    Builds upon cached note detection results to generate staff lines and
    quantize note positions. Results are cached based on all upstream
    parameters plus staff-specific settings.

    Args:
        image_id: Unique identifier for the registered image.
        threshold: Binarization threshold from binary processing stage.
        min_area: Minimum acceptable contour area from detection stage.
        max_area: Maximum acceptable contour area from detection stage.
        min_aspect: Minimum acceptable aspect ratio from detection stage.
        max_aspect: Maximum acceptable aspect ratio from detection stage.
        method: Staff fitting method ("Original", "Average", or "Adjustable").
        num_lines: Number of staff lines to generate.
        height_factor: Height adjustment factor for "Adjustable" method.

    Returns:
        StaffResult object containing staff lines and quantized notes.
    """
    # Get cached detection
    detection_result = cached_note_detection(
        image_id, threshold, min_area, max_area, min_aspect, max_aspect
    )

    # Create params and process
    params = StaffParams(
        method=method,
        num_lines=num_lines,
        height_factor=height_factor,
    )
    return create_staff(detection_result, params)


# Stage 4: MIDI Generation
@lru_cache(maxsize=MIDI_CACHE_SIZE)
def cached_midi_generation(
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
    map_to_scale: bool,
    scale_key: str,
    scale_type: str,
    quantize_rhythm: bool,
    grid_size: float,
    strength: float,
):
    """Cached version of MIDI generation with musical transformations.

    Builds upon cached staff creation results to generate final MIDI output.
    This is the most computationally expensive stage and has the smallest
    cache size due to the large parameter space.

    Args:
        image_id: Unique identifier for the registered image.
        threshold: Binarization threshold from binary processing stage.
        min_area: Minimum acceptable contour area from detection stage.
        max_area: Maximum acceptable contour area from detection stage.
        min_aspect: Minimum acceptable aspect ratio from detection stage.
        max_aspect: Maximum acceptable aspect ratio from detection stage.
        method: Staff fitting method from staff creation stage.
        num_lines: Number of staff lines from staff creation stage.
        height_factor: Height adjustment factor from staff creation stage.
        base_midi: Base MIDI note number for pitch mapping.
        tempo_bpm: Tempo in beats per minute.
        map_to_scale: Whether to apply scale mapping transformation.
        scale_key: Root key for scale mapping (if enabled).
        scale_type: Scale type for mapping (if enabled).
        quantize_rhythm: Whether to apply rhythm quantization.
        grid_size: Rhythmic grid size for quantization (if enabled).
        strength: Quantization strength (if rhythm quantization enabled).

    Returns:
        MidiResult object containing MIDI events and file data.
    """
    # Get cached staff result
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

    # Create params and process
    params = MidiParams(
        base_midi_note=base_midi,
        tempo_bpm=tempo_bpm,
        map_to_scale=map_to_scale,
        scale_key=scale_key,
        scale_type=scale_type,
        quantize_rhythm=quantize_rhythm,
        grid_size=grid_size,
        quantize_strength=strength,
    )
    return generate_midi(staff_result, params)


# Cache clear function
def clear_all_caches() -> None:
    """Clear all pipeline stage caches.

    Clears the LRU caches for all four pipeline stages. This is useful
    when memory usage becomes a concern or when you want to force
    recomputation of all results.
    """
    cached_binary_processing.cache_clear()
    cached_note_detection.cache_clear()
    cached_staff_creation.cache_clear()
    cached_midi_generation.cache_clear()
