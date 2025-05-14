"""Models for visualization outputs."""

import numpy as np
from pydantic import BaseModel


class VisualizationSet(BaseModel):
    """Complete set of visualizations for the UI."""

    binary_mask: np.ndarray | None = None
    note_detection: np.ndarray | None = None
    note_detection_binary: np.ndarray | None = None
    staff_lines: np.ndarray | None = None
    quantized_notes: np.ndarray | None = None
    piano_roll: np.ndarray | None = None

    class Config:
        arbitrary_types_allowed = True
