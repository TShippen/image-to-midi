"""
Musical transformation utilities for MIDI data.

This module provides functions for transforming MIDI events according to
musical principles like scales, key signatures, and rhythm quantization.
"""

import music21
import re

from image_to_midi.models import MidiEvent


# Discover available scales at module import time
AVAILABLE_SCALES = []
for attr_name in dir(music21.scale):
    if attr_name.endswith("Scale") and attr_name not in [
        "ConcreteScale",
        "AbstractScale",
    ]:
        try:
            scale_class = getattr(music21.scale, attr_name)
            # Try to instantiate with 'C' as the tonic
            scale_instance = scale_class("C")

            if hasattr(scale_instance, "getScaleDegreeFromPitch"):
                # Convert CamelCase to space-separated words for UI display
                display_name = re.sub(
                    r"([A-Z])", r" \1", attr_name.replace("Scale", "")
                ).strip()

                # Store as (display_name, class_name, class_object) for completeness
                AVAILABLE_SCALES.append((display_name, attr_name, scale_class))
        except (ValueError, TypeError, AttributeError):
            # Skip scales that can't be instantiated or don't have the right methods
            pass

# Sort for UI display
AVAILABLE_SCALES.sort(key=lambda x: x[0])

# For easy lookup: mapping from UI names to scale classes
SCALE_NAME_TO_CLASS = {name.lower(): cls for name, _, cls in AVAILABLE_SCALES}


def get_available_scale_names() -> list[str]:
    """Get a list of available scale types for UI display.

    Returns:
        List of user-friendly scale names
    """
    return [ui_name for ui_name, _, _ in AVAILABLE_SCALES]


def get_available_key_signatures() -> list[str]:
    """Get a list of common key signatures.

    Returns:
        List of key names (e.g., 'C', 'F#', 'Bb')
    """
    return ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]


def transpose_events(events: list[MidiEvent], semitones: int) -> list[MidiEvent]:
    """Transpose all MIDI events by a given number of semitones.

    Args:
        events: List of MidiEvent objects
        semitones: Number of semitones to transpose (positive or negative)

    Returns:
        List of transposed MidiEvent objects
    """
    if not events or semitones == 0:
        return events

    transposed = []
    for event in events:
        new_note = event.note + semitones
        # Ensure notes stay in valid MIDI range (0-127)
        new_note = max(0, min(127, new_note))

        transposed.append(event.model_copy(update={"note": new_note}))

    return transposed


def map_to_scale(
    events: list[MidiEvent], scale_key: str = "C", scale_name: str = "Major"
) -> list[MidiEvent]:
    """Map notes to the closest notes in a given scale.

    Args:
        events: List of MidiEvent objects
        scale_key: Root note of the scale (e.g., "C", "F#", "Bb")
        scale_name: UI display name of the scale (e.g., "Major", "Minor")

    Returns:
        List of MidiEvent objects with notes mapped to the scale
    """
    if not events:
        return events

    # Standardize scale name by converting to lowercase
    scale_name_lower = scale_name.lower()

    # Look up the scale class from our mapping
    found_scale_class = SCALE_NAME_TO_CLASS.get(scale_name_lower)

    # Default to MajorScale if not found
    if found_scale_class is None:
        found_scale_class = music21.scale.MajorScale

    # Create the scale instance
    sc = found_scale_class(scale_key)

    # Get all scale pitches for mapping
    scale_notes = []
    for midi_num in range(128):  # MIDI range 0-127
        p = music21.pitch.Pitch()
        p.midi = midi_num
        try:
            if sc.getScaleDegreeFromPitch(p) is not None:
                scale_notes.append(midi_num)
        except (ValueError, AttributeError):
            # Skip any pitches that cause errors with scale degree calculations
            pass

    # Map notes to the scale
    mapped_events = []
    for event in events:
        # If note already in scale, keep it
        if event.note in scale_notes:
            mapped_events.append(event)
            continue

        # Find closest note in scale
        closest_note = min(scale_notes, key=lambda n: abs(n - event.note))

        mapped_events.append(event.model_copy(update={"note": closest_note}))

    return mapped_events


def quantize_rhythm(
    events: list[MidiEvent],
    ticks_per_beat: int = 480,
    grid_size: float = 0.25,  # 1/4 beat = sixteenth note at 4/4
    strength: float = 1.0,  # 0.0 to 1.0
) -> list[MidiEvent]:
    """Quantize MIDI events to a rhythmic grid.

    Args:
        events: List of MidiEvent objects
        ticks_per_beat: Number of ticks per quarter note
        grid_size: Grid size as a fraction of a beat (e.g., 0.25 = 16th note)
        strength: Quantization strength (0.0 = none, 1.0 = full)

    Returns:
        List of MidiEvent objects with quantized timing
    """
    if not events or strength <= 0:
        return events

    # Calculate grid in ticks
    grid_ticks = int(ticks_per_beat * grid_size)

    quantized = []
    for event in events:
        # Calculate closest grid point for start time
        grid_start = round(event.start_tick / grid_ticks) * grid_ticks

        # Apply quantization with strength factor
        new_start = int(event.start_tick * (1 - strength) + grid_start * strength)

        # Also quantize duration if needed
        grid_dur = max(round(event.duration_tick / grid_ticks) * grid_ticks, grid_ticks)
        new_dur = int(event.duration_tick * (1 - strength) + grid_dur * strength)

        quantized.append(
            event.model_copy(update={"start_tick": new_start, "duration_tick": new_dur})
        )

    return sorted(quantized, key=lambda e: e.start_tick)


def get_key_name(midi_note: int) -> str:
    """Convert a MIDI note number to a key name (e.g., 'C4').

    Args:
        midi_note: MIDI note number

    Returns:
        Key name with octave
    """
    # Use music21's built-in conversion
    p = music21.pitch.Pitch()
    p.midi = midi_note
    return p.nameWithOctave
