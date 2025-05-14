import numpy as np
from image_to_midi.visualization import (
    create_binary_visualization,
    create_note_detection_visualizations,
    create_staff_visualization,
    create_piano_roll_visualization,
    create_detection_visualizations,
    create_staff_result_visualizations,
    create_all_visualizations,
)
from image_to_midi.models.pipeline_models import (
    BinaryResult,
    DetectionResult,
    StaffResult,
)
from image_to_midi.models.visualization_models import VisualizationSet
from image_to_midi.models.core_models import NoteBox


def test_create_binary_visualization_none():
    assert create_binary_visualization(None) is None


def test_create_binary_visualization_valid(simple_binary_blob):
    viz = create_binary_visualization(simple_binary_blob)
    assert viz.shape[2] == 3


def test_create_note_detection_visualizations_invalid():
    rgb, binv = create_note_detection_visualizations(None, None, [])
    assert rgb is None and binv is None


def test_create_note_detection_visualizations_valid(
    small_rgb_image, simple_binary_blob
):
    v1, v2 = create_note_detection_visualizations(
        small_rgb_image[..., ::-1], simple_binary_blob, [(10, 10, 5, 5)]
    )
    assert v1 is not None and v2 is not None


def test_create_staff_visualization_invalid():
    assert create_staff_visualization((10, 10), [], None) is None


def test_create_staff_visualization_valid():
    nb = [NoteBox(x=1, y=1, w=2, h=2)]
    lines = np.array([0, 5, 10])
    viz = create_staff_visualization((10, 10), nb, lines)
    assert viz.shape == (10, 10, 3)


def test_create_piano_roll_empty():
    img = create_piano_roll_visualization([])
    assert img.shape == (20, 200, 3)


def test_create_piano_roll_nonempty():
    from image_to_midi.models.core_models import MidiEvent

    evts = [
        MidiEvent(note=60, start_tick=0, duration_tick=10),
        MidiEvent(note=61, start_tick=10, duration_tick=5),
    ]
    img = create_piano_roll_visualization(evts)
    # height=(61-60+1)*10=20, width=int(15*1.1)=16
    assert img.shape == (20, 16, 3)


def test_create_detection_visualizations_invalid():
    rgb, binv = create_detection_visualizations(None, BinaryResult(), DetectionResult())
    assert rgb is None and binv is None


def test_create_staff_result_visualizations_invalid():
    orig, quant = create_staff_result_visualizations((10, 10), StaffResult())
    assert orig is None and quant is None


def test_create_all_visualizations_none():
    vs = create_all_visualizations(None, None, None, None, None)
    assert isinstance(vs, VisualizationSet)
