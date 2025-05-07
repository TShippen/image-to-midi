import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.image_processing import preprocess_image, create_note_visualization
from src.note_detection import detect_notes
from src.staff import (
    detect_lines,
    average_blob_height,
    adjust_blob_height,
    quantize_notes,
    calculate_fit_accuracy,
    calculate_note_variation,
    create_staff_visualization,
)
from src.midi_utils import build_note_events, write_midi_file, create_piano_roll

st.set_page_config(layout="wide")
st.title("Paint Splatter → Music Note Converter")
st.write("Transform wall paint splatters into musical representations")

# Initialize session state
for key in [
    "binary",
    "threshold",
    "notes",
    "min_area",
    "max_area",
    "min_aspect_ratio",
    "max_aspect_ratio",
    "lines",
    "quantized_notes",
    "line_method",
    "num_lines",
    "height_factor",
]:
    if key not in st.session_state:
        st.session_state[key] = None

# Sidebar
st.sidebar.header("Upload & Settings")
uploaded = st.sidebar.file_uploader(
    "Choose an image of paint splatters", type=["jpg", "jpeg", "png"]
)

if uploaded:
    try:
        pil = Image.open(uploaded)
    except Exception as e:
        st.sidebar.error(f"Could not read image: {e}")
        st.stop()

    st.sidebar.image(pil, caption="Original", use_column_width=True)
    arr = np.array(pil)
    orig = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    img_shape = orig.shape[:2]

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. Preprocessing",
            "2. Note Detection",
            "3. Staff Lines",
            "4. MIDI Generation",
        ]
    )

    # --- Tab 1 ---
    with tab1:
        st.header("Image Preprocessing")
        threshold = st.slider("Threshold", 0, 255, 93)

        @st.cache_data(show_spinner=False)
        def _prep(img: np.ndarray, thr: int) -> np.ndarray:
            return preprocess_image(img, thr)

        binary = _prep(orig, threshold)

        c1, c2 = st.columns(2)
        with c1:
            st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), caption="Original")
        with c2:
            st.image(binary, caption=f"Binary (thr={threshold})")

        if st.button("Confirm Preprocessing"):
            st.session_state.binary = binary
            st.session_state.threshold = threshold
            st.success("✅ Saved")

    # --- Tab 2 ---
    with tab2:
        st.header("Note Detection")
        if st.session_state.binary is None:
            st.warning("Complete preprocessing first")
        else:
            c1, c2 = st.columns(2)
            with c1:
                min_area = st.slider("Min Area", 0.01, 10.0, 1.0, 0.01)
                min_ar = st.slider("Min Aspect Ratio", 0.1, 5.0, 0.1, 0.01)
            with c2:
                max_area = st.slider("Max Area", 100, 5000, 5000, 100)
                max_ar = st.slider("Max Aspect Ratio", 5.0, 50.0, 20.0, 0.5)

            @st.cache_data(show_spinner=False)
            def _detect(bi, a, b, c, d):
                return detect_notes(bi, a, b, c, d)

            notes = _detect(st.session_state.binary, min_area, max_area, min_ar, max_ar)
            rgb_vis, bin_vis = create_note_visualization(
                orig, st.session_state.binary, notes
            )

            c1, c2 = st.columns(2)
            with c1:
                st.image(rgb_vis, caption="Detected on Original")
            with c2:
                st.image(bin_vis, caption=f"Detected on Binary (count={len(notes)})")

            st.info(f"Notes detected: {len(notes)}")
            if st.button("Confirm Note Detection"):
                st.session_state.notes = notes
                st.session_state.min_area = min_area
                st.session_state.max_area = max_area
                st.session_state.min_aspect_ratio = min_ar
                st.session_state.max_aspect_ratio = max_ar
                st.success("✅ Saved")

    # --- Tab 3 ---
    with tab3:
        st.header("Staff Line Creation")
        if st.session_state.notes is None:
            st.warning("Complete note detection first")
        else:
            method = st.radio(
                "Method", ["Original", "Average", "Adjustable"], horizontal=True
            )
            c1, c2 = st.columns(2)
            with c1:
                num_lines = st.slider("Number of Lines", 2, 50, 10)
            with c2:
                height_factor = (
                    st.slider("Height Factor", 0.0, 1.0, 0.5, 0.01)
                    if method == "Adjustable"
                    else 0.5
                )

            if method == "Original":
                lines = detect_lines(st.session_state.notes, num_lines)
                notes_adj = st.session_state.notes
            elif method == "Average":
                notes_adj = average_blob_height(st.session_state.notes)
                lines = detect_lines(notes_adj, num_lines)
            else:
                notes_adj = adjust_blob_height(st.session_state.notes, height_factor)
                lines = detect_lines(notes_adj, num_lines)

            acc = calculate_fit_accuracy(notes_adj, lines)
            var = calculate_note_variation(notes_adj, lines)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Fit Accuracy", f"{acc:.2f}%")
            with c2:
                st.metric("Note Variation", f"{var:.2f}")

            viz_orig = create_staff_visualization(img_shape, notes_adj, lines)
            quant = quantize_notes(notes_adj, lines)

            c1, c2 = st.columns(2)
            with c1:
                st.image(viz_orig, caption=f"{method} Method")
            with c2:
                st.image(
                    create_staff_visualization(img_shape, quant), caption="Quantized"
                )

            if st.button("Confirm Staff Lines"):
                st.session_state.lines = lines
                st.session_state.quantized_notes = quant
                st.session_state.line_method = method
                st.session_state.num_lines = num_lines
                st.session_state.height_factor = height_factor
                st.success("✅ Saved")

    # --- Tab 4 ---
    with tab4:
        st.header("MIDI Generation")
        if st.session_state.quantized_notes is None:
            st.warning("Complete staff creation first")
        else:
            c1, c2 = st.columns(2)
            with c1:
                base = st.slider("Base MIDI Note", 21, 108, 60)
                name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                octave = base // 12 - 1
                st.caption(f"{name[base % 12]}{octave}")
            with c2:
                tempo = st.slider("Tempo (BPM)", 30, 240, 120)

            events = build_note_events(
                st.session_state.quantized_notes,
                st.session_state.lines,
                img_shape[1],
                base,
            )
            with st.spinner("Generating piano roll…"):
                roll = create_piano_roll(
                    events, img_shape[1], len(st.session_state.lines)
                )
            st.image(roll, caption="Piano Roll", use_column_width=True)

            with st.spinner("Composing MIDI…"):
                midi_bytes = write_midi_file(events, tempo)
            st.audio(midi_bytes, format="audio/midi")

            st.download_button(
                "Download MIDI",
                data=midi_bytes,
                file_name="paint_to_music.mid",
                mime="audio/midi",
            )
