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
    """Return an RGB piano-roll image for a sequence of MidiEvent models."""
    if not events:  # nothing to draw
        return np.ones((20, 200, 3), np.uint8) * 255

    # time & pitch bounds ---------------------------------------------------
    end_tick = max(e.start_tick + e.duration_tick for e in events)
    lowest = min(e.note for e in events)
    highest = max(e.note for e in events)

    pitch_span = highest - lowest + 1
    height = pitch_span * 10  # 10 px per semitone
    width = int(end_tick * 1.1)  # 10 % right-hand margin

    img = np.ones((height, width, 3), np.uint8) * 255

    # draw horizontal key-lines (white keys darker)
    for i in range(pitch_span + 1):
        y = i * 10
        key_colour = (
            (150, 150, 150)
            if (lowest + i) % 12 in [0, 2, 4, 5, 7, 9, 11]
            else (200, 200, 200)
        )
        cv2.line(img, (0, y), (width, y), key_colour, 1)

    # draw each note block
    for ev in events:
        y0 = (ev.note - lowest) * 10
        top, bot = height - y0 - 10, height - y0
        left, right = ev.start_tick, ev.start_tick + ev.duration_tick

        hsv = np.array([[[((ev.note % 12) / 12) * 180, 200, 200]]], np.uint8)
        fill = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0].tolist()

        cv2.rectangle(img, (left, top), (right, bot), fill, -1)
        cv2.rectangle(img, (left, top), (right, bot), (0, 0, 0), 1)

    return img
