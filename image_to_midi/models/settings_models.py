"""Parameter models for pipeline configuration.

This module defines Pydantic models that encapsulate all configurable
parameters for each stage of the image-to-MIDI conversion pipeline.
These models provide validation, default values, and clear interfaces
for customizing the behavior of each processing step.
"""

from pydantic import BaseModel, Field


class ImageProcessingParams(BaseModel):
    """Configuration parameters for image preprocessing and binarization.

    Controls the conversion of input images to binary masks suitable for
    note detection. The threshold parameter determines the cutoff point
    for converting grayscale pixels to binary values.

    Attributes:
        threshold: Grayscale threshold for binarization (0-255, default 93).
    """

    threshold: int = Field(
        93, ge=0, le=255, description="Grayscale threshold for binarization"
    )


class NoteDetectionParams(BaseModel):
    """Configuration parameters for note detection via contour analysis.

    Controls the filtering criteria used to identify valid musical notes
    from detected contours in the binary image. Parameters define the
    acceptable size and shape characteristics of note-like objects.

    Attributes:
        min_area: Minimum contour area in pixels (default 1.0).
        max_area: Maximum contour area in pixels (default 5000.0).
        min_aspect_ratio: Minimum width/height ratio (default 0.1).
        max_aspect_ratio: Maximum width/height ratio (default 20.0).
    """

    min_area: float = Field(1.0, ge=0.01, description="Minimum contour area in pixels")
    max_area: float = Field(
        5000.0, ge=100.0, description="Maximum contour area in pixels"
    )
    min_aspect_ratio: float = Field(
        0.1, ge=0.1, description="Minimum width/height ratio"
    )
    max_aspect_ratio: float = Field(
        20.0, ge=5.0, description="Maximum width/height ratio"
    )


class StaffParams(BaseModel):
    """Configuration parameters for staff line generation and note quantization.

    Controls how staff lines are positioned and how detected notes are
    quantized to those staff positions. Different methods provide trade-offs
    between preserving original note characteristics and creating consistent
    staff spacing.

    Attributes:
        method: Staff fitting method ("Original", "Average", or "Adjustable").
        num_lines: Number of staff lines to generate (2-50, default 10).
        height_factor: Height adjustment factor for "Adjustable" method (0.0-1.0).
    """

    method: str = Field("Original", description="Staff fitting method")
    num_lines: int = Field(10, ge=2, le=50, description="Number of staff lines")
    height_factor: float = Field(
        0.5, ge=0.0, le=1.0, description="Height adjustment factor"
    )


class MidiParams(BaseModel):
    """Configuration parameters for MIDI generation and musical transformations.

    Controls the conversion of quantized notes to MIDI events and applies
    various musical transformations such as scale mapping, rhythm quantization,
    and harmonic additions. Provides extensive customization of the final
    musical output.

    Attributes:
        base_midi_note: Base MIDI note number for staff mapping (21-108, default 60).
        tempo_bpm: Tempo in beats per minute (30-240, default 120).
        transpose_semitones: Transposition in semitones (-36 to +36, default 0).
        map_to_scale: Whether to map notes to a specific musical scale.
        scale_key: Root key for scale mapping (default "C").
        scale_type: Type of scale for mapping (default "major").
        quantize_rhythm: Whether to quantize note timing to a rhythmic grid.
        grid_size: Rhythmic grid size as fraction of beat (0.0625-1.0, default 0.25).
        quantize_strength: Strength of rhythm quantization (0.0-1.0, default 1.0).
        add_harmony: Whether to add harmonic accompaniment (experimental).
        chord_type: Type of chords to add (default "triad").
        voice_leading: Whether to use voice leading in harmony (default True).
    """

    base_midi_note: int = Field(
        60, ge=21, le=108, description="Base MIDI note for staff mapping"
    )
    tempo_bpm: int = Field(120, ge=30, le=240, description="Tempo in beats per minute")

    transpose_semitones: int = Field(
        0, ge=-36, le=36, description="Transposition in semitones"
    )

    map_to_scale: bool = Field(False, description="Enable scale mapping")
    scale_key: str = Field("C", description="Root key for scale mapping")
    scale_type: str = Field("major", description="Scale type for mapping")

    quantize_rhythm: bool = Field(False, description="Enable rhythm quantization")
    grid_size: float = Field(
        0.25, ge=0.0625, le=1.0, description="Rhythmic grid size as fraction of beat"
    )
    quantize_strength: float = Field(
        1.0, ge=0.0, le=1.0, description="Rhythm quantization strength"
    )

    add_harmony: bool = Field(False, description="Enable harmonic accompaniment")
    chord_type: str = Field("triad", description="Type of chords to add")
    voice_leading: bool = Field(True, description="Use voice leading in harmony")


class ProcessingParameters(BaseModel):
    """Complete configuration for the entire image-to-MIDI pipeline.

    Aggregates all parameter sets for every stage of the conversion process,
    providing a single object that can be passed to the main pipeline function.
    Each component uses sensible defaults but can be customized as needed.

    Attributes:
        image: Parameters for image preprocessing and binarization.
        detection: Parameters for note detection and filtering.
        staff: Parameters for staff line generation and quantization.
        midi: Parameters for MIDI generation and musical transformations.
    """

    image: ImageProcessingParams = Field(
        default_factory=ImageProcessingParams, description="Image processing parameters"
    )
    detection: NoteDetectionParams = Field(
        default_factory=NoteDetectionParams, description="Note detection parameters"
    )
    staff: StaffParams = Field(
        default_factory=StaffParams, description="Staff generation parameters"
    )
    midi: MidiParams = Field(
        default_factory=MidiParams, description="MIDI generation parameters"
    )
