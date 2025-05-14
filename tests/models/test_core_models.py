import pytest
from pydantic import ValidationError
from image_to_midi.models import NoteBox, MidiEvent


def test_notebox_properties(valid_notebox):
    assert valid_notebox.cx == pytest.approx(1 + 3 / 2)
    assert valid_notebox.cy == pytest.approx(2.0 + 4.0 / 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x": -1, "y": 0.0, "w": 1, "h": 1.0},
        {"x": 0, "y": -0.1, "w": 1, "h": 1.0},
        {"x": 0, "y": 0.0, "w": 0, "h": 1.0},
        {"x": 0, "y": 0.0, "w": 1, "h": 0.9},
    ],
)
def test_notebox_validation_raises(kwargs):
    with pytest.raises(ValidationError):
        NoteBox(**kwargs)


def test_midievent_valid(valid_midievent):
    assert valid_midievent.note == 60
    assert valid_midievent.start_tick == 0
    assert valid_midievent.duration_tick == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"note": -1, "start_tick": 0, "duration_tick": 1},
        {"note": 128, "start_tick": 0, "duration_tick": 1},
        {"note": 60, "start_tick": -1, "duration_tick": 1},
        {"note": 60, "start_tick": 0, "duration_tick": 0},
    ],
)
def test_midievent_validation_raises(kwargs):
    with pytest.raises(ValidationError):
        MidiEvent(**kwargs)
