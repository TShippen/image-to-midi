"""Models for visualization outputs.

This module defines data structures for holding visualization results
from each stage of the image-to-MIDI pipeline. The VisualizationSet
model aggregates all visual outputs in a single container, making
it easy to pass visualization data to the user interface.
"""

import numpy as np
from pydantic import BaseModel, Field


class VisualizationSet(BaseModel):
    """Complete set of visualizations for the user interface.

    Aggregates all visual outputs from the image-to-MIDI pipeline into
    a single container. Each field represents a different visualization
    type that can be displayed in the UI, allowing users to inspect
    the results of each processing stage.

    Attributes:
        binary_mask: Binary image mask from preprocessing, or None if unavailable.
        note_detection: RGB image with detected note boxes overlaid, or None.
        note_detection_binary: Binary image with detected note boxes overlaid, or None.
        staff_lines: Visualization of staff lines with original notes, or None.
        quantized_notes: Visualization of staff lines with quantized notes, or None.
        piano_roll: Piano roll visualization of MIDI events, or None.
    """

    binary_mask: np.ndarray | None = Field(
        None, description="Binary image mask from preprocessing"
    )
    note_detection: np.ndarray | None = Field(
        None, description="RGB image with detected note boxes"
    )
    note_detection_binary: np.ndarray | None = Field(
        None, description="Binary image with detected note boxes"
    )
    staff_lines: np.ndarray | None = Field(
        None, description="Staff lines with original notes"
    )
    quantized_notes: np.ndarray | None = Field(
        None, description="Staff lines with quantized notes"
    )
    piano_roll: np.ndarray | None = Field(
        None, description="Piano roll visualization of MIDI events"
    )

    class Config:
        arbitrary_types_allowed = True
