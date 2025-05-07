# src/models.py
from pydantic import BaseModel, Field


class NoteBox(BaseModel):
    """Axis-aligned bounding box around one paint splatter."""

    x: int = Field(..., ge=0)  # left
    y: float = Field(..., ge=0)  # top  (float keeps life simple later)
    w: int = Field(..., ge=1)  # width  in px
    h: float = Field(..., ge=1.0)  # height in px

    # convenience
    @property
    def cx(self) -> float:  # centre-x
        return self.x + self.w / 2

    @property
    def cy(self) -> float:  # centre-y
        return self.y + self.h / 2


class MidiEvent(BaseModel):
    """One MIDI note; time is measured in ticks."""

    note: int = Field(..., ge=0, le=127)
    start_tick: int = Field(..., ge=0)
    duration_tick: int = Field(..., ge=1)
