import pytest

from image_to_midi.music_transformations import (
    get_available_scale_names,
    get_available_key_signatures,
    transpose_events,
    map_to_scale,
    quantize_rhythm,
    get_key_name,
)
from image_to_midi.models import MidiEvent


def test_available_scales_not_empty() -> None:
    names = get_available_scale_names()
    assert "Major" in names


def test_key_signatures_order() -> None:
    keys = get_available_key_signatures()
    assert keys[:3] == ["C", "G", "D"]


def test_transpose_events_no_change() -> None:
    events = [MidiEvent(note=60, start_tick=0, duration_tick=1)]
    result = transpose_events(events, 0)
    # should return the same list object when semitones == 0
    assert result is events


def test_transpose_events_clamp_high() -> None:
    ev = MidiEvent(note=127, start_tick=0, duration_tick=1)
    result = transpose_events([ev], 5)
    # 127 + 5 would overflow, so it clamps back to 127
    assert result[0].note == 127


def test_map_to_scale_identity_member() -> None:
    ev = MidiEvent(note=60, start_tick=0, duration_tick=1)
    mapped = map_to_scale([ev], scale_key="C", scale_name="Major")
    # 60 is already in C-Major, so it stays
    assert mapped[0].note == 60


def test_map_to_scale_non_member_snaps() -> None:
    ev = MidiEvent(note=61, start_tick=0, duration_tick=1)
    mapped = map_to_scale([ev], scale_key="C", scale_name="Major")
    # 61 (C#) should snap to either C (60) or D (62)
    assert mapped[0].note in (60, 62)


@pytest.mark.parametrize("strength", [0, 0.0])
def test_quantize_rhythm_strength_zero(strength) -> None:
    events = [MidiEvent(note=60, start_tick=123, duration_tick=45)]
    out = quantize_rhythm(events, ticks_per_beat=100, grid_size=0.5, strength=strength)
    # strength ≤ 0 should skip quantization entirely
    assert out is events


def test_quantize_rhythm_full_strength() -> None:
    events = [MidiEvent(note=60, start_tick=123, duration_tick=45)]
    out = quantize_rhythm(events, ticks_per_beat=100, grid_size=0.5, strength=1)
    # grid size = 0.5 of 100 ticks → 50‐tick grid
    assert out[0].start_tick % 50 == 0
    assert out[0].duration_tick % 50 == 0


def test_get_key_name() -> None:
    key = get_key_name(60)
    assert key.startswith("C4")
