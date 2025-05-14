"""Models for representing pipeline processing stages."""

import numpy as np
from pydantic import BaseModel, Field

from image_to_midi.models.core_models import NoteBox, MidiEvent


class BinaryResult(BaseModel):
    """Result of the image binarization stage."""

    binary_mask: np.ndarray | None = None

    class Config:
        arbitrary_types_allowed = True


class DetectionResult(BaseModel):
    """Results from the note detection stage."""

    note_boxes: list[NoteBox] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class StaffResult(BaseModel):
    """Staff line and note quantization data."""

    lines: np.ndarray = Field(default_factory=lambda: np.array([]))
    original_boxes: list[NoteBox] = Field(default_factory=list)
    quantized_boxes: list[NoteBox] = Field(default_factory=list)
    fit_accuracy: float = 0.0
    pitch_variation: float = 0.0

    class Config:
        arbitrary_types_allowed = True


class MidiResult(BaseModel):
    """MIDI generation results."""

    events: list[MidiEvent] = Field(default_factory=list)
    midi_bytes: bytes | None = None
    midi_file_path: str = ""  # Path to temporary MIDI file
    base_note_name: str = ""

    class Config:
        arbitrary_types_allowed = True
