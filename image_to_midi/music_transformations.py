"""
Musical transformation utilities for MIDI data.

This module provides functions for transforming MIDI events according to
musical principles like scales, key signatures, and rhythm quantization.
"""

import logging
import re

import music21
from music21.scale import ConcreteScale

from image_to_midi.models import MidiEvent

logger = logging.getLogger(__name__)


# Discover only concrete, octave-repeating scales at import time
AVAILABLE_SCALES = []
for attr_name in dir(music21.scale):
    if not attr_name.endswith("Scale"):
        continue
    scale_class = getattr(music21.scale, attr_name)
    if not isinstance(scale_class, type) or not issubclass(scale_class, ConcreteScale):
        continue
    try:
        inst = scale_class("C")
        if not hasattr(inst, "getScaleDegreeFromPitch"):
            continue
        display_name = re.sub(
            r"([A-Z])", r" \1", attr_name.replace("Scale", "")
        ).strip()
        AVAILABLE_SCALES.append((display_name, attr_name, scale_class))
    except (TypeError, ValueError, AttributeError) as exception:
        # Something we anticipated (bad ctor args, missing method, etc.)
        logger.debug(f"Skipping scale {attr_name!r}: {exception}")
        continue

AVAILABLE_SCALES.sort(key=lambda x: x[0])
SCALE_NAME_TO_CLASS = {name.lower(): cls for name, _, cls in AVAILABLE_SCALES}


def get_available_scale_names() -> list[str]:
    """Get a list of available scale types for UI display.

    Returns:
        List of user-friendly scale names (e.g., ["Major", "Minor", ...]).
    """
    return [ui_name for ui_name, _, _ in AVAILABLE_SCALES]


def get_available_key_signatures() -> list[str]:
    """Get a list of common key signatures.

    Returns:
        List of key names in standard order (e.g., ["C", "G", "D", ...]).
    """
    return ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]


def transpose_events(events: list[MidiEvent], semitones: int) -> list[MidiEvent]:
    """Transpose all MIDI events by a given number of semitones.

    Args:
        events: MIDI events to transpose.
        semitones: Number of semitones to transpose (positive or negative).

    Returns:
        A new list of MidiEvent objects with each note shifted by `semitones`,
        clamped to the valid MIDI range 0–127.
    """
    if not events or semitones == 0:
        return events

    result = []
    for event in events:
        note = event.note + semitones
        note = max(0, min(127, note))
        result.append(event.model_copy(update={"note": note}))
    return result


def map_to_scale(
    events: list[MidiEvent], scale_key: str = "C", scale_name: str = "Major"
) -> list[MidiEvent]:
    """Map notes to the closest notes in a given scale (across all octaves).

    Args:
        events: MIDI events to map.
        scale_key: Root note of the scale (e.g., "C", "F#", "Bb").
        scale_name: Display name of the scale (e.g., "Major", "Minor").

    Returns:
        A new list of MidiEvent objects where each note is replaced by the
        nearest pitch in the specified scale, searching across the full
        MIDI range (0–127).
    """
    if not events:
        return events

    scale_cls = SCALE_NAME_TO_CLASS.get(scale_name.lower(), music21.scale.MajorScale)
    scale_obj = scale_cls(scale_key)

    scale_notes = []
    for midi_num in range(128):
        p = music21.pitch.Pitch()
        p.midi = midi_num
        try:
            if scale_obj.getScaleDegreeFromPitch(p) is not None:
                scale_notes.append(midi_num)
        except (ValueError, AttributeError):
            continue

    mapped = []
    for event in events:
        if event.note in scale_notes:
            mapped.append(event)
        else:
            closest = min(scale_notes, key=lambda n: abs(n - event.note))
            mapped.append(event.model_copy(update={"note": closest}))
    return mapped


def quantize_rhythm(
    events: list[MidiEvent],
    ticks_per_beat: int = 480,
    grid_size: float = 0.25,
    strength: float = 1.0,
) -> list[MidiEvent]:
    """Quantize MIDI events to a rhythmic grid.

    Args:
        events: MIDI events to quantize.
        ticks_per_beat: Number of ticks per quarter note (default 480).
        grid_size: Grid size as a fraction of a beat (e.g., 0.25 = 16th note).
        strength: Quantization strength, from 0.0 (none) to 1.0 (full).

    Returns:
        A new list of MidiEvent objects with start times and durations
        moved toward the nearest grid lines by the given strength, sorted
        by start tick.
    """
    if not events or strength <= 0:
        return events

    grid_ticks = int(ticks_per_beat * grid_size)
    quantized = []
    for event in events:
        grid_start = round(event.start_tick / grid_ticks) * grid_ticks
        new_start = int(event.start_tick * (1 - strength) + grid_start * strength)

        grid_dur = max(round(event.duration_tick / grid_ticks) * grid_ticks, grid_ticks)
        new_dur = int(event.duration_tick * (1 - strength) + grid_dur * strength)

        quantized.append(
            event.model_copy(update={"start_tick": new_start, "duration_tick": new_dur})
        )

    return sorted(quantized, key=lambda e: e.start_tick)


def get_key_name(midi_note: int) -> str:
    """Convert a MIDI note number to a key name (e.g., "C4").

    Args:
        midi_note: MIDI note number (0–127).

    Returns:
        The pitch name with octave (e.g., "C4", "G#3").
    """
    p = music21.pitch.Pitch()
    p.midi = midi_note
    return p.nameWithOctave
