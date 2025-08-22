"""Models for representing pipeline processing stages.

This module contains Pydantic models that encapsulate the results of each
stage in the image-to-MIDI conversion pipeline. Each model represents the
output data from a specific processing step, enabling clean separation of
concerns and easy testing of individual pipeline components.
"""

import numpy as np
from pydantic import BaseModel, Field

from image_to_midi.models.core_models import NoteBox, MidiEvent


class BinaryResult(BaseModel):
    """Result of the image binarization stage.

    Contains the output of converting an input image to a binary mask
    suitable for note detection. The binary mask isolates regions of
    interest (typically dark paint splatters or musical notation) as
    white pixels on a black background.

    Attributes:
        binary_mask: 2D numpy array of binary image data, or None if processing failed.
    """

    binary_mask: np.ndarray | None = Field(
        None, description="Binary mask from image processing"
    )

    class Config:
        arbitrary_types_allowed = True


class DetectionResult(BaseModel):
    """Results from the note detection stage.

    Contains the bounding boxes of detected musical notes or paint splatters
    found in the binary image. Each detected region is represented as a
    NoteBox with position and size information.

    Attributes:
        note_boxes: List of detected note bounding boxes, empty if no notes found.
    """

    note_boxes: list[NoteBox] = Field(
        default_factory=list, description="Detected note bounding boxes"
    )

    class Config:
        arbitrary_types_allowed = True


class StaffResult(BaseModel):
    """Staff line detection and note quantization results.

    Contains the generated staff lines and the results of quantizing detected
    notes to those staff positions. Includes both the original detected notes
    and their quantized versions, along with quality metrics.

    Attributes:
        lines: 1D numpy array of staff line y-positions, empty if generation failed.
        original_boxes: Original detected note boxes before quantization.
        quantized_boxes: Note boxes after quantization to staff lines.
        fit_accuracy: Percentage of notes that don't overlap staff lines (0-100).
        pitch_variation: Mean standard deviation of note positions within staff slots.
    """

    lines: np.ndarray = Field(
        default_factory=lambda: np.array([]), description="Staff line y-positions"
    )
    original_boxes: list[NoteBox] = Field(
        default_factory=list, description="Original detected note boxes"
    )
    quantized_boxes: list[NoteBox] = Field(
        default_factory=list, description="Quantized note boxes"
    )
    fit_accuracy: float = Field(0.0, description="Percentage of non-overlapping notes")
    pitch_variation: float = Field(
        0.0, description="Mean standard deviation of note positions"
    )

    class Config:
        arbitrary_types_allowed = True


class MidiResult(BaseModel):
    """MIDI generation and synthesis results.

    Contains the final MIDI data generated from quantized notes, including
    both the structured event data and the serialized MIDI file format.
    Also includes file paths for playback and display purposes.

    Attributes:
        events: List of MIDI note events with timing and pitch information.
        midi_bytes: Serialized MIDI file data, or None if generation failed.
        midi_file_path: Path to temporary MIDI file for playback purposes.
        base_note_name: Human-readable name of the base MIDI note (e.g., "C4").
    """

    events: list[MidiEvent] = Field(
        default_factory=list, description="Generated MIDI note events"
    )
    midi_bytes: bytes | None = Field(None, description="Serialized MIDI file data")
    midi_file_path: str = Field("", description="Path to temporary MIDI file")
    base_note_name: str = Field("", description="Human-readable base note name")

    class Config:
        arbitrary_types_allowed = True
