import io
import mido
import numpy as np
from mido import MidiFile, MidiTrack, Message, MetaMessage

from image_to_midi.models import MidiEvent, NoteBox


def build_note_events(
    boxes: list[NoteBox],
    line_pos: np.ndarray,
    base_midi: int = 60,
    ticks_per_px: int = 4,
) -> list[MidiEvent]:

    def closest(y: float) -> int:
        return int(np.argmin(np.abs(line_pos - y)))

    events: list[MidiEvent] = []
    for b in boxes:
        idx = closest(b.cy)
        note = base_midi + (len(line_pos) - idx - 1)
        start = int(b.x * ticks_per_px)
        dur = max(int(b.w * ticks_per_px), 1)
        events.append(MidiEvent(note=note, start_tick=start, duration_tick=dur))
    return sorted(events, key=lambda e: e.start_tick)


def write_midi_file(
    events: list[MidiEvent], tempo_bpm: int = 120, ticks_per_beat: int = 480
) -> bytes:
    """Compose raw MIDI bytes from MidiEvent models."""
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))

    timeline: list[tuple[str, int, int]] = []
    for e in events:
        timeline.append(("on", e.start_tick, e.note))
        timeline.append(("off", e.start_tick + e.duration_tick, e.note))
    timeline.sort(key=lambda x: x[1])

    prev = 0
    for kind, tick, note in timeline:
        delta = tick - prev
        msg = "note_on" if kind == "on" else "note_off"
        track.append(Message(msg, note=note, velocity=64, time=delta))
        prev = tick

    buf = io.BytesIO()
    mid.save(file=buf)
    buf.seek(0)
    return buf.getvalue()
