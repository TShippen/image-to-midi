"""Core domain models for image-to-midi conversion."""

from pydantic import BaseModel, Field


class NoteBox(BaseModel):
    """Axis-aligned bounding box representing a detected musical note.

    This model represents a rectangular region in an image that contains
    a detected musical note or paint splatter. The coordinates follow
    standard computer vision conventions with (0,0) at the top-left.

    Attributes:
        x: Left edge position in pixels (non-negative integer).
        y: Top edge position in pixels (non-negative float).
        w: Width in pixels (positive integer).
        h: Height in pixels (positive float).
    """

    x: int = Field(..., ge=0, description="Left edge position in pixels")
    y: float = Field(..., ge=0, description="Top edge position in pixels")
    w: int = Field(..., ge=1, description="Width in pixels")
    h: float = Field(..., ge=1.0, description="Height in pixels")

    @property
    def cx(self) -> float:
        """Calculate the horizontal center coordinate of the bounding box.

        Returns:
            The x-coordinate of the box center.
        """
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        """Calculate the vertical center coordinate of the bounding box.

        Returns:
            The y-coordinate of the box center.
        """
        return self.y + self.h / 2


class MidiEvent(BaseModel):
    """A single MIDI note event with timing information.

    Represents a musical note in MIDI format with pitch and timing data.
    Time is measured in MIDI ticks, which can be converted to actual time
    based on the tempo and ticks-per-beat settings of the MIDI file.

    Attributes:
        note: MIDI note number (0-127, where 60 is middle C).
        start_tick: Start time in MIDI ticks (non-negative).
        duration_tick: Duration in MIDI ticks (positive).
    """

    note: int = Field(..., ge=0, le=127, description="MIDI note number (0-127)")
    start_tick: int = Field(..., ge=0, description="Start time in MIDI ticks")
    duration_tick: int = Field(..., ge=1, description="Duration in MIDI ticks")
