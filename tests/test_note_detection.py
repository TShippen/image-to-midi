import numpy as np
import cv2
from src.note_detection import detect_notes

def test_detect_notes_empty():
    empty = np.zeros((50, 50), dtype=np.uint8)
    notes = detect_notes(empty, 1.0, 100.0, 0.1, 10.0)
    assert notes == []

def test_detect_notes_simple():
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (30, 30), 255, -1)
    notes = detect_notes(img, 100, 1000, 0.5, 2.0)
    assert len(notes) == 1
    x, y, w, h = notes[0]
    assert x == 10 and y == 10
    assert w == 21 and h == 21
