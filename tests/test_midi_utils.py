import numpy as np
from image_to_midi.midi_utils import build_note_events, write_midi_file
from image_to_midi.models.core_models import NoteBox


def test_build_note_events_order_and_pitch():
    lines = np.array([0, 10, 20])
    boxes = [NoteBox(x=0, y=0, w=1, h=1), NoteBox(x=5, y=15, w=2, h=2)]
    events = build_note_events(boxes, lines, base_midi=60, ticks_per_px=1)
    # First event starts at x=0, second at x=5
    assert events[0].start_tick == 0
    assert events[1].start_tick == 5
    # y=0→idx=0→note=60+len(lines)-1-0=62
    assert events[0].note == 62


def test_write_midi_file_empty():
    data = write_midi_file([], tempo_bpm=120, ticks_per_beat=480)
    assert isinstance(data, (bytes, bytearray))
    # Should at least contain 'MThd'
    assert b"MThd" in data
