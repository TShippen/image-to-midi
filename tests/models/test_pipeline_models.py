import numpy as np
from image_to_midi.models.pipeline_models import (
    BinaryResult,
    DetectionResult,
    StaffResult,
    MidiResult,
)


def test_binaryresult_defaults():
    b = BinaryResult()
    assert b.binary_mask is None


def test_detectionresult_defaults():
    d = DetectionResult()
    assert d.note_boxes == []


def test_staffresult_defaults():
    s = StaffResult()
    assert isinstance(s.lines, np.ndarray)
    assert s.lines.size == 0
    assert s.original_boxes == []
    assert s.quantized_boxes == []
    assert s.fit_accuracy == 0.0
    assert s.pitch_variation == 0.0


def test_midiresult_defaults():
    m = MidiResult()
    assert m.events == []
    assert m.midi_bytes is None
    assert m.midi_file_path == ""
    assert m.base_note_name == ""


def test_pipeline_models_arbitrary_types():
    # Ensure numpy arrays and bytes accepted
    b = BinaryResult(binary_mask=np.zeros((1, 1)))
    assert b.binary_mask.shape == (1, 1)
    m = MidiResult(midi_bytes=b"abc")
    assert m.midi_bytes == b"abc"
