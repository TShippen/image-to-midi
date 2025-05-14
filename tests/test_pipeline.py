from image_to_midi.pipeline import (
    process_binary_image,
    detect_notes,
    create_staff,
    generate_midi,
    process_complete_pipeline,
)
from image_to_midi.models.settings_models import (
    ImageProcessingParams,
    NoteDetectionParams,
    StaffParams,
    MidiParams,
)
from image_to_midi.models.pipeline_models import (
    BinaryResult,
    DetectionResult,
    StaffResult,
    MidiResult,
)


def test_process_binary_image_none():
    out = process_binary_image(None, ImageProcessingParams())
    assert isinstance(out, BinaryResult)
    assert out.binary_mask is None


def test_detect_notes_none():
    res = detect_notes(BinaryResult(), NoteDetectionParams())
    assert isinstance(res, DetectionResult)
    assert res.note_boxes == []


def test_create_staff_empty():
    staff = create_staff(DetectionResult(), StaffParams())
    assert isinstance(staff, StaffResult)
    assert staff.lines.size == 0


def test_generate_midi_empty():
    midi = generate_midi(StaffResult(), MidiParams())
    assert isinstance(midi, MidiResult)
    assert midi.events == []


def test_process_complete_pipeline_none():
    br, dr, sr, mr, vs = process_complete_pipeline(
        None,
        ImageProcessingParams(),
        NoteDetectionParams(),
        StaffParams(),
        MidiParams(),
    )
    assert isinstance(br, BinaryResult)
    assert isinstance(dr, DetectionResult)
    assert isinstance(sr, StaffResult)
    assert isinstance(mr, MidiResult)
