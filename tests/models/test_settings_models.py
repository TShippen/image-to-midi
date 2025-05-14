import pytest
from pydantic import ValidationError
from image_to_midi.models import (
    ImageProcessingParams,
    NoteDetectionParams,
    StaffParams,
    MidiParams,
    ProcessingParameters,
)


def test_imageprocessingparams_default():
    p = ImageProcessingParams()
    assert 0 <= p.threshold <= 255


@pytest.mark.parametrize("val", [-1, 256])
def test_imageprocessingparams_threshold_invalid(val):
    with pytest.raises(ValidationError):
        ImageProcessingParams(threshold=val)


def test_notedetectionparams_default():
    p = NoteDetectionParams()
    assert p.min_area < p.max_area
    assert p.min_aspect_ratio < p.max_aspect_ratio


def test_staffparams_default():
    p = StaffParams()
    assert 2 <= p.num_lines <= 50
    assert 0.0 <= p.height_factor <= 1.0


def test_midiparams_default():
    p = MidiParams()
    assert 21 <= p.base_midi_note <= 108
    assert 30 <= p.tempo_bpm <= 240


def test_processingparameters_contains_all():
    p = ProcessingParameters()
    assert hasattr(p, "image")
    assert hasattr(p, "detection")
    assert hasattr(p, "staff")
    assert hasattr(p, "midi")
