"""MIDI generation and audio synthesis utilities.

This module provides functions for converting detected and quantized notes
into MIDI format and synthesizing audio from MIDI data. It handles the
conversion from image coordinates to musical timing and pitch, generates
standard MIDI files, and provides audio synthesis capabilities.
"""

import io
import logging
import os
from functools import lru_cache

import mido
import pretty_midi
import soundfile as sf
import numpy as np
from mido import MidiFile, MidiTrack, Message, MetaMessage

from image_to_midi.models import MidiEvent, NoteBox

logger = logging.getLogger(__name__)


def build_note_events(
    boxes: list[NoteBox],
    line_pos: np.ndarray,
    base_midi: int = 60,
    ticks_per_px: int = 4,
) -> list[MidiEvent]:
    """Convert quantized note boxes to MIDI events with pitch and timing.

    Maps the vertical position of each note box to a MIDI pitch based on
    staff line positions, and converts horizontal position and width to
    MIDI timing information. The resulting events are sorted by start time.

    Args:
        boxes: List of note boxes to convert to MIDI events.
        line_pos: 1D array of staff line y-positions for pitch mapping.
        base_midi: Base MIDI note number for the bottom staff line (default 60).
        ticks_per_px: Conversion factor from pixels to MIDI ticks (default 4).

    Returns:
        List of MidiEvent objects sorted by start_tick, or empty list if
        no boxes provided or line_pos is empty.
    """

    def closest(y: float) -> int:
        """Find the index of the closest staff line to the given y-coordinate."""
        return int(np.argmin(np.abs(line_pos - y)))

    events: list[MidiEvent] = []
    for box in boxes:
        # Find closest staff line and calculate MIDI note number
        line_idx = closest(box.cy)
        note_number = base_midi + (len(line_pos) - line_idx - 1)

        # Convert position and width to MIDI timing
        start_tick = int(box.x * ticks_per_px)
        duration_tick = max(int(box.w * ticks_per_px), 1)

        events.append(
            MidiEvent(
                note=note_number, start_tick=start_tick, duration_tick=duration_tick
            )
        )

    return sorted(events, key=lambda e: e.start_tick)


def write_midi_file(
    events: list[MidiEvent], tempo_bpm: int = 120, ticks_per_beat: int = 480
) -> bytes:
    """Generate a MIDI file from a list of MIDI events.

    Creates a standard MIDI file with a single track containing all the
    provided events. Sets the specified tempo and converts events to
    proper MIDI note_on/note_off message pairs with correct timing.

    Args:
        events: List of MidiEvent objects to include in the file.
        tempo_bpm: Tempo in beats per minute (default 120).
        ticks_per_beat: MIDI ticks per quarter note (default 480).

    Returns:
        MIDI file data as bytes, suitable for writing to a .mid file
        or loading in a MIDI player.
    """
    # Create MIDI file with single track
    midi_file = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    midi_file.tracks.append(track)

    # Set tempo at the beginning of the track
    track.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))

    # Create timeline of all note on/off events
    timeline: list[tuple[str, int, int]] = []
    for event in events:
        timeline.append(("on", event.start_tick, event.note))
        timeline.append(("off", event.start_tick + event.duration_tick, event.note))

    # Sort events by time
    timeline.sort(key=lambda x: x[1])

    # Convert to MIDI messages with delta times
    previous_tick = 0
    for event_type, tick, note in timeline:
        delta_time = tick - previous_tick
        message_type = "note_on" if event_type == "on" else "note_off"
        track.append(Message(message_type, note=note, velocity=64, time=delta_time))
        previous_tick = tick

    # Serialize to bytes
    buffer = io.BytesIO()
    midi_file.save(file=buffer)
    buffer.seek(0)
    return buffer.getvalue()


@lru_cache(maxsize=None)
def midi_to_audio(midi_path: str) -> str | None:
    """Synthesize a MIDI file to audio using software synthesis.

    Converts a MIDI file to a WAV audio file using the pretty_midi library's
    built-in software synthesizer. The output is normalized to prevent clipping
    and cached for improved performance on repeated requests.

    Args:
        midi_path: Path to the input MIDI file to synthesize.

    Returns:
        Path to the generated WAV file, or None if synthesis failed.
        The WAV file will have the same base name as the input MIDI file.

    Note:
        Results are cached indefinitely using lru_cache to avoid redundant
        synthesis of the same MIDI file.
    """
    try:
        # Load MIDI file and synthesize to audio
        pretty_midi_obj = pretty_midi.PrettyMIDI(midi_path)
        audio_data = pretty_midi_obj.synthesize(fs=44100)

        # Normalize audio to prevent clipping
        peak_amplitude = np.max(np.abs(audio_data))
        if peak_amplitude > 0:
            audio_data = audio_data / peak_amplitude

        # Save as WAV file with same base name
        wav_path = os.path.splitext(midi_path)[0] + ".wav"
        sf.write(wav_path, audio_data, 44100)

        return wav_path

    except Exception as e:
        logger.error(f"Failed to synthesize MIDI to audio: {e}")
        return None
