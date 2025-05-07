import numpy as np
from image_to_midi.midi_utils import (
    build_note_events,
    write_midi_file,
    create_piano_roll,
)


def test_build_note_events():
    blobs = [(0, 0, 10, 10)]
    lines = np.array([0.0, 20.0])
    events = build_note_events(blobs, lines, base_midi_note=60, ticks_per_px=1)
    assert isinstance(events, list)
    note, start, dur = events[0]
    assert note == 61
    assert start == 0
    assert dur == 10


def test_write_midi_and_piano_roll():
    events = [(60, 0, 10), (61, 10, 5)]
    midi = write_midi_file(events, tempo_bpm=120, ticks_per_beat=480)
    assert isinstance(midi, (bytes, bytearray))

    roll = create_piano_roll(events, img_width=100, line_count=2)
    assert roll.ndim == 3
    assert roll.dtype == np.uint8
