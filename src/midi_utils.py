import io
import mido
import numpy as np
from mido import MidiFile, MidiTrack, Message, MetaMessage
import cv2


def build_note_events(
    blobs: list[tuple[int, float, int, float]],
    line_positions: np.ndarray,
    img_width: int,
    base_midi_note: int = 60,
    ticks_per_px: int = 4,
) -> list[tuple[int, int, int]]:
    """Map blobs to sorted MIDI note events.

    Args:
        blobs: List of (x, y, w, h).
        line_positions: 1D array of staff Y-positions.
        img_width: Width of original image.
        base_midi_note: MIDI note for bottom line.
        ticks_per_px: Resolution scale.

    Returns:
        List of (note, start_tick, duration_tick).
    """

    def closest_line(y: float) -> int:
        deltas = np.abs(line_positions - y)
        return int(np.argmin(deltas))

    events: list[tuple[int, int, int]] = []
    num_lines = len(line_positions)
    for x, y, w, h in blobs:
        cy = y + h / 2
        idx = closest_line(cy)
        note = base_midi_note + (num_lines - idx - 1)
        start = int(x * ticks_per_px)
        dur = max(int(w * ticks_per_px), 1)
        events.append((note, start, dur))
    return sorted(events, key=lambda e: e[1])


def write_midi_file(
    events: list[tuple[int, int, int]], tempo_bpm: int = 120, ticks_per_beat: int = 480
) -> bytes:
    """Compose a MIDI file from note events.

    Args:
        events: (note, start_tick, duration_tick).
        tempo_bpm: Tempo in BPM.
        ticks_per_beat: PPQ resolution.

    Returns:
        Raw MIDI bytes.
    """
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)
    tempo = mido.bpm2tempo(tempo_bpm)
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))

    timeline: list[tuple[str, int, int]] = []
    for note, st, du in events:
        timeline.append(("on", st, note))
        timeline.append(("off", st + du, note))
    timeline.sort(key=lambda x: x[1])

    prev = 0
    for kind, tick, note in timeline:
        delta = tick - prev
        msg_type = "note_on" if kind == "on" else "note_off"
        track.append(Message(msg_type, note=note, velocity=64, time=delta))
        prev = tick

    buf = io.BytesIO()
    mid.save(file=buf)
    buf.seek(0)
    return buf.getvalue()


def create_piano_roll(
    events: list[tuple[int, int, int]], img_width: int, line_count: int
) -> np.ndarray:
    """Draw a simple piano-roll image from MIDI events.

    Args:
        events: (note, start_tick, duration_tick).
        img_width: Width for time scaling (unused).
        line_count: Number of staff lines → pitch span.

    Returns:
        H×W×3 uint8 RGB numpy array.
    """
    end = max((s + d) for _, s, d in events) if events else 1
    notes = [n for n, _, _ in events]
    low, high = min(notes), max(notes) if notes else (0, 0)
    span = high - low + 1
    height = span * 10
    width = int(end * 1.1)

    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    # draw key lines
    for i in range(span + 1):
        y = i * 10
        color = (200, 200, 200)
        if (low + i) % 12 in [0, 2, 4, 5, 7, 9, 11]:
            color = (150, 150, 150)
        cv2.line(canvas, (0, y), (width, y), color, 1)

    for note, start, dur in events:
        y0 = (note - low) * 10
        cv2.rectangle(
            canvas,
            (start, height - y0 - 10),
            (start + dur, height - y0),
            tuple(
                int(c)
                for c in cv2.cvtColor(
                    np.array([[[((note % 12) / 12) * 180, 200, 200]]], np.uint8),
                    cv2.COLOR_HSV2RGB,
                )[0][0]
            ),
            -1,
        )
        cv2.rectangle(
            canvas, (start, height - y0 - 10), (start + dur, height - y0), (0, 0, 0), 1
        )
    return canvas
