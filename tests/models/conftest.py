import pytest
from image_to_midi.models import NoteBox, MidiEvent


@pytest.fixture
def valid_notebox():
    return NoteBox(x=1, y=2.0, w=3, h=4.0)


@pytest.fixture
def valid_midievent():
    return MidiEvent(note=60, start_tick=0, duration_tick=1)
