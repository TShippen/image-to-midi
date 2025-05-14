import numpy as np
from image_to_midi.note_detection import detect_notes


def test_detect_notes_no_contours():
    empty = np.zeros((50, 50), dtype=np.uint8)
    boxes = detect_notes(empty, 1, 100, 0.1, 10)
    assert boxes == []


def test_detect_notes_single_blob(simple_binary_blob):
    boxes = detect_notes(
        simple_binary_blob,
        min_area=10,
        max_area=500,
        min_aspect_ratio=0.5,
        max_aspect_ratio=2,
    )
    # Our square is 21×21 pixels, aspect=1
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert (x, y) == (10, 10)
    assert (w, h) == (21, 21)
