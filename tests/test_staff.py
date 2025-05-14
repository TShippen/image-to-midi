import numpy as np
import pytest

from image_to_midi.staff import (
    detect_lines,
    average_box_height,
    adjust_box_height,
    vertical_quantize_notes,
    horizontal_quantize_notes,
    calculate_fit_accuracy,
    calculate_note_variation,
)
from image_to_midi.models import NoteBox


def make_boxes():
    return [NoteBox(x=0, y=0, h=2.0, w=2), NoteBox(x=0, y=10, h=4.0, w=2)]


def test_detect_lines():
    boxes = make_boxes()
    lines = detect_lines(boxes, num_lines=3)
    assert len(lines) == 3
    assert lines[0] == pytest.approx(0.0)


def test_average_box_height():
    boxes = make_boxes()
    out = average_box_height(boxes)
    avg = np.mean([2.0, 4.0])
    assert all(b.h == avg for b in out)


def test_adjust_box_height():
    boxes = make_boxes()
    out = adjust_box_height(boxes, factor=0.5)
    # factor=0.5 → target halfway between min=2 and max=4 → 3.0
    assert all(b.h == 3.0 for b in out)


def test_vertical_quantize_notes(line_positions):
    # one box at y halfway → should map into slot 0≤cy<Ly1
    boxes = [NoteBox(x=0, y=line_positions[0], w=1, h=1)]
    out = vertical_quantize_notes(boxes, line_positions)
    assert len(out) == 1


def test_horizontal_quantize_notes():
    boxes = [NoteBox(x=3, y=0, w=4, h=1)]
    out = horizontal_quantize_notes(boxes, image_width=16, grid_divisions=4, strength=1)
    assert isinstance(out[0].x, int)
    assert out[0].w >= 1


def test_calculate_fit_accuracy_and_variation():
    # two boxes, one overlaps line exactly
    lines = np.array([0, 5, 10])
    boxes = [NoteBox(x=0, y=0, h=5, w=1), NoteBox(x=0, y=6, h=2, w=1)]
    acc = calculate_fit_accuracy(boxes, lines)
    var = calculate_note_variation(boxes, lines)
    assert 0.0 <= acc <= 100.0
    assert isinstance(var, float)
