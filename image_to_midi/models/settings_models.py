"""Parameter models for pipeline configuration."""

from pydantic import BaseModel, Field


class ImageProcessingParams(BaseModel):
    """Parameters for image processing stage."""

    threshold: int = Field(93, ge=0, le=255)


class NoteDetectionParams(BaseModel):
    """Parameters for note detection stage."""

    min_area: float = Field(1.0, ge=0.01)
    max_area: float = Field(5000.0, ge=100.0)
    min_aspect_ratio: float = Field(0.1, ge=0.1)
    max_aspect_ratio: float = Field(20.0, ge=5.0)


class StaffParams(BaseModel):
    """Parameters for staff line detection and note quantization."""

    method: str = Field("Original")
    num_lines: int = Field(10, ge=2, le=50)
    height_factor: float = Field(0.5, ge=0.0, le=1.0)


class MidiParams(BaseModel):
    """Parameters for MIDI generation."""

    base_midi_note: int = Field(60, ge=21, le=108)
    tempo_bpm: int = Field(120, ge=30, le=240)

    # Musical transformation parameters
    transpose_semitones: int = Field(0, ge=-36, le=36)

    map_to_scale: bool = Field(False)
    scale_key: str = Field("C")
    scale_type: str = Field("major")

    quantize_rhythm: bool = Field(False)
    grid_size: float = Field(0.25, ge=0.0625, le=1.0)  # From 64th note to quarter note
    quantize_strength: float = Field(1.0, ge=0.0, le=1.0)

    add_harmony: bool = Field(False)
    chord_type: str = Field("triad")
    voice_leading: bool = Field(True)


class ProcessingParameters(BaseModel):
    """Complete set of parameters for the entire pipeline."""

    image: ImageProcessingParams = Field(default_factory=ImageProcessingParams)
    detection: NoteDetectionParams = Field(default_factory=NoteDetectionParams)
    staff: StaffParams = Field(default_factory=StaffParams)
    midi: MidiParams = Field(default_factory=MidiParams)
