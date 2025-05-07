"""
Streamlit UI for “Paint-splatter → MIDI” after the full Pydantic refactor.

Key domain models
-----------------
* NoteBox  – bounding box around one paint splatter (src.models)
* MidiEvent – one MIDI note (src.models)

Data flow
---------
1.  Preprocess → binary mask
2.  Detect splatter boxes  → raw tuples **and** NoteBox models
3.  Staff-line fitting     → list[NoteBox]
4.  Quantise to staff      → list[NoteBox]
5.  MIDI events            → list[MidiEvent]
6.  Piano-roll & download
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.image_processing import preprocess_image, create_note_visualization
from src.note_detection import detect_notes
from src.staff import (
    detect_lines,
    average_box_height,
    adjust_box_height,
    quantize_notes,
    calculate_fit_accuracy,
    calculate_note_variation,
    create_staff_visualization,
)
from src.midi_utils import build_note_events, write_midi_file, create_piano_roll
from src.models import NoteBox

# -----------------------------------------------------------------------------
# Streamlit page setup
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("Paint Splatter → Music Note Converter")
st.write("Upload a wall-paint splatter photo and turn its shapes into music!")

# -----------------------------------------------------------------------------
# Ensure every session_state key exists (so we can use `is None` checks later)
# -----------------------------------------------------------------------------
for key in (
    "binary",
    "threshold",
    "raw_boxes",  # list[tuple[int,int,int,int]]
    "note_boxes",  # list[NoteBox]
    "min_area",
    "max_area",
    "min_aspect",
    "max_aspect",
    "lines",  # np.ndarray
    "quantized_boxes",  # list[NoteBox]
    "line_method",
    "num_lines",
    "height_factor",
):
    st.session_state.setdefault(key, None)

# -----------------------------------------------------------------------------
# Sidebar – file upload
# -----------------------------------------------------------------------------
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image of paint splatters", type=("jpg", "jpeg", "png")
)

# -----------------------------------------------------------------------------
# If we have an image, continue with the workflow
# -----------------------------------------------------------------------------
if uploaded_file:
    # -------------------------------------------------------------------------
    # Load & display the original image
    # -------------------------------------------------------------------------
    try:
        pil_img = Image.open(uploaded_file)
    except Exception as err:
        st.sidebar.error(f"Could not read image: {err}")
        st.stop()

    st.sidebar.image(pil_img, caption="Original", use_column_width=True)

    rgb_arr = np.array(pil_img)  # RGB (H,W,3)
    bgr_img = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    img_h, img_w = bgr_img.shape[:2]

    # -------------------------------------------------------------------------
    # Workflow tabs
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["1. Preprocess", "2. Detect Notes", "3. Staff Lines", "4. MIDI"]
    )

    # ---------------------------------------------------------------------
    # TAB 1 – preprocessing
    # ---------------------------------------------------------------------
    with tab1:
        st.header("1 Preprocessing")
        threshold_val = st.slider("Threshold", 0, 255, 93)

        @st.cache_data(show_spinner=False)
        def _preprocess(img: np.ndarray, thr: int) -> np.ndarray:
            return preprocess_image(img, thr)

        binary_mask = _preprocess(bgr_img, threshold_val)

        c1, c2 = st.columns(2)
        c1.image(rgb_arr, caption="Original")
        c2.image(binary_mask, caption=f"Binary (thr {threshold_val})")

        if st.button("Confirm preprocessing"):
            st.session_state.binary = binary_mask
            st.session_state.threshold = threshold_val
            st.success("✅ Preprocessing saved")

    # ---------------------------------------------------------------------
    # TAB 2 – detect bounding boxes
    # ---------------------------------------------------------------------
    with tab2:
        st.header("2 Note detection")
        if st.session_state.binary is None:
            st.warning("Run the preprocessing step first.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                min_area = st.slider("Min area", 0.01, 10.0, 1.0, 0.01)
                min_aspect = st.slider("Min aspect-ratio", 0.1, 5.0, 0.1, 0.01)
            with c2:
                max_area = st.slider("Max area", 100, 5_000, 5_000, 100)
                max_aspect = st.slider("Max aspect-ratio", 5.0, 50.0, 20.0, 0.5)

            @st.cache_data(show_spinner=False)
            def _detect(
                mask: np.ndarray,
                area_min: float,
                area_max: float,
                ar_min: float,
                ar_max: float,
            ) -> list[tuple[int, int, int, int]]:
                return detect_notes(mask, area_min, area_max, ar_min, ar_max)

            raw_boxes = _detect(
                st.session_state.binary, min_area, max_area, min_aspect, max_aspect
            )

            # convert to NoteBox models for later steps
            note_boxes = [
                NoteBox(x=x, y=float(y), w=w, h=float(h)) for x, y, w, h in raw_boxes
            ]

            # visualise detections (needs raw tuples)
            vis_rgb, vis_bin = create_note_visualization(
                bgr_img, st.session_state.binary, raw_boxes
            )

            vc1, vc2 = st.columns(2)
            vc1.image(vis_rgb, caption="Detections on original")
            vc2.image(vis_bin, caption=f"Detections on binary (count {len(raw_boxes)})")

            st.info(f"{len(raw_boxes)} note-like shapes found")

            if st.button("Confirm detections"):
                st.session_state.raw_boxes = raw_boxes
                st.session_state.note_boxes = note_boxes
                st.session_state.min_area = min_area
                st.session_state.max_area = max_area
                st.session_state.min_aspect = min_aspect
                st.session_state.max_aspect = max_aspect
                st.success("✅ Detections saved")

    # ---------------------------------------------------------------------
    # TAB 3 – fit staff lines & quantise
    # ---------------------------------------------------------------------
    with tab3:
        st.header("3 Staff-line creation")
        if st.session_state.note_boxes is None:
            st.warning("Please detect notes first.")
        else:
            method = st.radio(
                "Line fitting method",
                ("Original", "Average", "Adjustable"),
                horizontal=True,
            )

            c1, c2 = st.columns(2)
            num_lines = c1.slider("Number of lines", 2, 50, 10)
            height_factor = (
                c2.slider("Height factor", 0.0, 1.0, 0.5, 0.01)
                if method == "Adjustable"
                else 0.5
            )

            # choose a working list of NoteBox
            if method == "Original":
                working_boxes = st.session_state.note_boxes
            elif method == "Average":
                working_boxes = average_box_height(st.session_state.note_boxes)
            else:  # Adjustable
                working_boxes = adjust_box_height(
                    st.session_state.note_boxes, height_factor
                )

            lines = detect_lines(working_boxes, num_lines)
            quantised = quantize_notes(working_boxes, lines)

            acc = calculate_fit_accuracy(working_boxes, lines)
            var = calculate_note_variation(working_boxes, lines)

            m1, m2 = st.columns(2)
            m1.metric("Fit accuracy", f"{acc:.2f}%")
            m2.metric("Pitch variation", f"{var:.2f}")

            viz_orig = create_staff_visualization((img_h, img_w), working_boxes, lines)
            viz_quant = create_staff_visualization((img_h, img_w), quantised, lines)

            iv1, iv2 = st.columns(2)
            iv1.image(viz_orig, caption=f"{method} lines")
            iv2.image(viz_quant, caption="Quantised to staff")

            if st.button("Confirm staff-line settings"):
                st.session_state.lines = lines
                st.session_state.quantized_boxes = quantised
                st.session_state.line_method = method
                st.session_state.num_lines = num_lines
                st.session_state.height_factor = height_factor
                st.success("✅ Staff-line settings saved")

    # ---------------------------------------------------------------------
    # TAB 4 – build MIDI, piano roll, download
    # ---------------------------------------------------------------------
    with tab4:
        st.header("4 MIDI generation")
        if st.session_state.quantized_boxes is None:
            st.warning("Quantise notes first.")
        else:
            lc1, lc2 = st.columns(2)
            base_midi = lc1.slider("Base MIDI note", 21, 108, 60)
            tempo_bpm = lc2.slider("Tempo (BPM)", 30, 240, 120)

            note_names = [
                "C",
                "C#",
                "D",
                "D#",
                "E",
                "F",
                "F#",
                "G",
                "G#",
                "A",
                "A#",
                "B",
            ]
            lc1.caption(f"Base note {note_names[base_midi % 12]}{base_midi // 12 - 1}")

            # build MidiEvent list
            events = build_note_events(
                st.session_state.quantized_boxes,
                st.session_state.lines,
                base_midi,
            )

            # piano-roll
            with st.spinner("Rendering piano roll…"):
                roll_img = create_piano_roll(events)
            st.image(
                roll_img, caption="Piano-roll visualisation", use_column_width=True
            )

            # MIDI bytes
            with st.spinner("Writing MIDI…"):
                midi_bytes = write_midi_file(events, tempo_bpm)

            st.audio(midi_bytes, format="audio/midi")
            st.download_button(
                label="Download MIDI",
                data=midi_bytes,
                file_name="paint_to_music.mid",
                mime="audio/midi",
            )
