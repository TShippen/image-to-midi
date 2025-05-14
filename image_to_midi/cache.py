# cache.py
"""Caching mechanisms for the image-to-MIDI pipeline."""

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
def cached_binary_processing(image_id, threshold):
    """Cache for binary image processing.

    Args:
        image_id: A unique identifier for the image (hashed)
        threshold: Binarization threshold

    Returns:
        BinaryResult object
    """
    # Get the image from your global storage
    from image_to_midi.app_state import get_image_by_id

    image = get_image_by_id(image_id)

    params = ImageProcessingParams(threshold=threshold)
    return process_binary_image(image, params)


# Stage 2: Note Detection
@lru_cache(maxsize=DETECTION_CACHE_SIZE)
def cached_note_detection(
    image_id, threshold, min_area, max_area, min_aspect, max_aspect
):
    """Cache for note detection.

    Returns:
        DetectionResult object
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
    """Cache for staff creation.

    Returns:
        StaffResult object
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
    map_to_scale,
    scale_key,
    scale_type,
    quantize_rhythm,
    grid_size,
    strength,
):
    """Cache for MIDI generation.

    Returns:
        MidiResult object
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
def clear_all_caches():
    """Clear all pipeline caches."""
    cached_binary_processing.cache_clear()
    cached_note_detection.cache_clear()
    cached_staff_creation.cache_clear()
    cached_midi_generation.cache_clear()
