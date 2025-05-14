import numpy as np
from image_to_midi.models.visualization_models import VisualizationSet


def test_visualizationset_defaults():
    v = VisualizationSet()
    # All fields default to None or empty array
    assert v.binary_mask is None
    assert v.note_detection is None
    assert v.piano_roll is None
