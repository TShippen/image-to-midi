import numpy as np
import cv2
import pytest

from image_to_midi.models.core_models import NoteBox
from image_to_midi.models.pipeline_models import MidiEvent


@pytest.fixture
def small_rgb_image():
    # 2×2 RGB image: red, green, blue, black
    img = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [0, 0, 0]]], dtype=np.uint8
    )
    return img


@pytest.fixture
def simple_binary_blob():
    # 100×100 binary mask with one square blob at (10,10)-(30,30)
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (30, 30), 255, -1)
    return mask


@pytest.fixture
def notebox_list():
    # Three NoteBoxes at distinct x,y
    return [
        NoteBox(x=0, y=0.0, w=10, h=5.0),
        NoteBox(x=5, y=20.0, w=8, h=4.0),
        NoteBox(x=15, y=40.0, w=5, h=3.0),
    ]


@pytest.fixture
def line_positions():
    # 5 equally spaced lines from y=0 to y=40
    return np.linspace(0, 40, 5)


@pytest.fixture
def midi_events():
    # Two simple MidiEvent instances
    return [
        MidiEvent(note=60, start_tick=0, duration_tick=10),
        MidiEvent(note=62, start_tick=10, duration_tick=5),
    ]
