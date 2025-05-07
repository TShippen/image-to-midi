import io
import mido
import numpy as np
from mido import MidiFile, MidiTrack, Message, MetaMessage
import cv2

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


def create_piano_roll(events: list[MidiEvent]) -> np.ndarray:
    """Draw a simple piano-roll image from MIDI events.

    Args:
        events: (note, start_tick, duration_tick).

    Returns:
        H×W×3 uint8 RGB numpy array.
    """
    end = max((s + d) for _, s, d in events) if events else 1
    notes = [n for n, _, _ in events]
    low, high = (min(notes), max(notes)) if notes else (0, 0)
    span = high - low + 1
    height = span * 10
    width = int(end * 1.1)

    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    # draw key lines
    for i in range(span + 1):
        y0 = i * 10
        color = (200, 200, 200)
        if (low + i) % 12 in [0, 2, 4, 5, 7, 9, 11]:
            color = (150, 150, 150)
        cv2.line(canvas, (0, y0), (width, y0), color, 1)

    # draw notes
    for note, start, dur in events:
        y0 = (note - low) * 10
        # rainbow fill via HSV→RGB
        hsv = np.array([[[((note % 12) / 12) * 180, 200, 200]]], np.uint8)
        fill = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0].tolist()

        top = height - y0 - 10
        bottom = height - y0
        cv2.rectangle(canvas, (start, top), (start + dur, bottom), fill, -1)
        cv2.rectangle(canvas, (start, top), (start + dur, bottom), (0, 0, 0), 1)

    return canvas
