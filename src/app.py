import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from midiutil import MIDIFile
import plotly.graph_objs as go
from skimage import measure, morphology, filters
import mido
import io
from PIL import Image

st.set_page_config(layout="wide")

st.title("Paint Splatter to Music Note Converter")
st.write("Transform wall paint splatters into musical representations")

# Sidebar for all controls
st.sidebar.header("Upload & Settings")

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose an image of paint splatters", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image in the sidebar
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="Original Image", use_column_width=True)

    # Convert to OpenCV format
    img_array = np.array(image)
    original_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_shape = original_image.shape[:2]

    # Create tabs for workflow steps
    tab1, tab2, tab3, tab4 = st.tabs(
        ["1. Image Preprocessing", "2. Note Detection", "3. Staff Line Creation", "4. MIDI Generation"])

    # Tab 1: Image Preprocessing
    with tab1:
        st.header("Image Preprocessing")

        # Thresholding controls
        threshold = st.slider("Threshold Value", 0, 255, 93, 1)


        # Image processing function (from your notebook)
        def preprocess_image(image, threshold_value):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
            return binary


        # Process and display
        binary = preprocess_image(original_image, threshold)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Original Image")
        with col2:
            st.image(binary, caption=f"Binary Image (Threshold: {threshold})")

        # Add button to confirm and proceed
        if st.button("Confirm Preprocessing", key="confirm_preprocessing"):
            st.session_state.binary = binary
            st.session_state.threshold = threshold
            st.success("✅ Preprocessing settings saved")

    # Tab 2: Note Detection
    with tab2:
        st.header("Note Detection")

        # Check if preprocessing is done
        if not hasattr(st.session_state, 'binary'):
            st.warning("Please complete the preprocessing step first")
        else:
            # Note detection controls
            col1, col2 = st.columns(2)
            with col1:
                min_area = st.slider("Minimum Area", 0.01, 10.0, 1.0, 0.01)
                min_aspect_ratio = st.slider("Min Aspect Ratio", 0.1, 5.0, 0.1, 0.01)
            with col2:
                max_area = st.slider("Maximum Area", 100, 5000, 5000, 100)
                max_aspect_ratio = st.slider("Max Aspect Ratio", 5.0, 50.0, 20.0, 0.5)


            # Note detection function (from your notebook)
            @st.cache_data
            def detect_notes(binary_img, min_area, max_area, min_aspect_ratio, max_aspect_ratio):
                contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                notes = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    area = cv2.contourArea(contour)
                    if min_area < area < max_area and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                        notes.append((x, y, w, h))
                return notes


            # Detect and visualize notes
            notes = detect_notes(st.session_state.binary, min_area, max_area, min_aspect_ratio, max_aspect_ratio)


            # Create a visualization similar to your notebook
            def create_note_visualization(img, binary, detected_notes):
                # Create RGB versions for drawing
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else cv2.cvtColor(img,
                                                                                                        cv2.COLOR_GRAY2RGB)
                binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

                # Draw rectangles on both images
                for note in detected_notes:
                    x, y, w, h = note
                    cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.rectangle(binary_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

                return img_rgb, binary_rgb


            img_with_notes, binary_with_notes = create_note_visualization(
                original_image, st.session_state.binary, notes
            )

            col1, col2 = st.columns(2)
            with col1:
                st.image(img_with_notes, caption="Original Image with Detected Notes")
            with col2:
                st.image(binary_with_notes, caption=f"Binary Image with Detected Notes (Count: {len(notes)})")

            st.info(f"Number of notes detected: {len(notes)}")

            # Button to confirm and proceed
            if st.button("Confirm Note Detection", key="confirm_notes"):
                st.session_state.notes = notes
                st.session_state.min_area = min_area
                st.session_state.max_area = max_area
                st.session_state.min_aspect_ratio = min_aspect_ratio
                st.session_state.max_aspect_ratio = max_aspect_ratio
                st.success("✅ Note detection settings saved")

    # Tab 3: Staff Line Creation
    with tab3:
        st.header("Staff Line Creation")

        # Check if note detection is done
        if not hasattr(st.session_state, 'notes'):
            st.warning("Please complete the note detection step first")
        else:
            # Line detection methods
            method = st.radio("Line Detection Method",
                              ["Original", "Average", "Adjustable"], horizontal=True)

            col1, col2 = st.columns(2)
            with col1:
                num_lines = st.slider("Number of Lines", 2, 50, 10, 1)

            with col2:
                if method == "Adjustable":
                    height_factor = st.slider("Height Factor", 0.0, 1.0, 0.5, 0.01)
                else:
                    height_factor = 0.5  # Default


            # Line detection functions
            def detect_lines_method1(notes, num_lines):
                y_values = [y + h / 2 for _, y, _, h in notes]
                top = min(y_values)
                bottom = max(y_values)
                return np.linspace(top, bottom, num_lines)


            def average_blob_height(notes):
                avg_height = np.mean([h for _, _, _, h in notes])
                averaged_notes = []
                for x, y, w, h in notes:
                    center_y = y + h / 2
                    new_y = center_y - avg_height / 2
                    averaged_notes.append((x, new_y, w, avg_height))
                return averaged_notes


            def adjust_blob_height(notes, height_factor):
                min_height = min(h for _, _, _, h in notes)
                max_height = max(h for _, _, _, h in notes)
                new_height = min_height + (max_height - min_height) * height_factor
                adjusted_notes = []
                for x, y, w, h in notes:
                    center_y = y + h / 2
                    new_y = center_y - new_height / 2
                    adjusted_notes.append((x, new_y, w, new_height))
                return adjusted_notes


            def quantize_notes(notes, lines):
                quantized_notes = []
                for x, y, w, h in notes:
                    center_y = y + h / 2
                    for i in range(len(lines) - 1):
                        if lines[i] <= center_y < lines[i + 1]:
                            new_center = (lines[i] + lines[i + 1]) / 2
                            new_height = lines[i + 1] - lines[i] - 1  # Leave a small gap
                            new_y = new_center - new_height / 2
                            quantized_notes.append((x, new_y, w, new_height))
                            break
                return quantized_notes


            def calculate_fit_accuracy(notes, lines):
                total_blobs = len(notes)
                overlapping_blobs = 0
                for _, y, _, h in notes:
                    blob_top = y
                    blob_bottom = y + h
                    for line in lines:
                        if blob_top <= line <= blob_bottom:
                            overlapping_blobs += 1
                            break
                return (total_blobs - overlapping_blobs) / total_blobs * 100


            def calculate_note_variation(notes, lines):
                note_groups = [[] for _ in range(len(lines) - 1)]
                for x, y, w, h in notes:
                    center_y = y + h / 2
                    for i in range(len(lines) - 1):
                        if lines[i] <= center_y < lines[i + 1]:
                            note_groups[i].append(center_y)
                            break

                variations = []
                for group in note_groups:
                    if len(group) > 1:
                        variations.append(np.std(group))

                return np.mean(variations) if variations else 0


            # Apply the selected method
            if method == "Original":
                lines = detect_lines_method1(st.session_state.notes, num_lines)
                notes_to_display = st.session_state.notes
            elif method == "Average":
                averaged_notes = average_blob_height(st.session_state.notes)
                lines = detect_lines_method1(averaged_notes, num_lines)
                notes_to_display = averaged_notes
            else:  # Adjustable
                adjusted_notes = adjust_blob_height(st.session_state.notes, height_factor)
                lines = detect_lines_method1(adjusted_notes, num_lines)
                notes_to_display = adjusted_notes

            # Calculate metrics
            accuracy = calculate_fit_accuracy(notes_to_display, lines)
            variation = calculate_note_variation(notes_to_display, lines)

            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fit Accuracy", f"{accuracy:.2f}%")
            with col2:
                st.metric("Note Variation", f"{variation:.2f}")


            # Visualization function
            def create_staff_visualization(img_shape, notes, lines, quantized=False):
                # Create a blank white image
                height, width = img_shape
                img = np.ones((height, width, 3), dtype=np.uint8) * 255

                # Draw the notes as black rectangles
                for x, y, w, h in notes:
                    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 0), -1)

                # Draw the lines as red lines
                for line in lines:
                    cv2.line(img, (0, int(line)), (width, int(line)), (255, 0, 0), 1)

                return img


            # Create visualizations
            quantized_notes = quantize_notes(notes_to_display, lines)
            original_viz = create_staff_visualization(img_shape, notes_to_display, lines)
            quantized_viz = create_staff_visualization(img_shape, quantized_notes, lines)

            col1, col2 = st.columns(2)
            with col1:
                st.image(original_viz, caption=f"Detected Notes with Lines ({method} Method)")
            with col2:
                st.image(quantized_viz, caption="Quantized Notes with Lines")

            # Button to confirm and proceed
            if st.button("Confirm Staff Line Settings", key="confirm_lines"):
                st.session_state.lines = lines
                st.session_state.quantized_notes = quantized_notes
                st.session_state.line_method = method
                st.session_state.num_lines = num_lines
                st.session_state.height_factor = height_factor
                st.success("✅ Staff line settings saved")

    # Tab 4: MIDI Generation
    with tab4:
        st.header("MIDI Generation and Visualization")

        # Check if staff line creation is done
        if not hasattr(st.session_state, 'quantized_notes'):
            st.warning("Please complete the staff line creation step first")
        else:
            # MIDI controls
            col1, col2 = st.columns(2)
            with col1:
                base_midi_note = st.slider("Base MIDI Note", 21, 108, 60, 1)
                st.caption(
                    f"Base Note: {['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][base_midi_note % 12]}{base_midi_note // 12 - 1}")
            with col2:
                tempo_bpm = st.slider("Tempo (BPM)", 30, 240, 120, 1)


            # MIDI generation functions
            def find_closest_line(y, line_positions):
                return np.argmin(np.abs(line_positions - y))


            def build_note_events(blobs, line_positions, img_width, base_midi_note=60, ticks_per_px=4):
                num_lines = len(line_positions)
                events = []

                for x, y, w, h in blobs:
                    center_y = y + h / 2
                    line_idx = find_closest_line(center_y, line_positions)
                    midi_note = base_midi_note + (num_lines - line_idx - 1)

                    start_tick = int(x * ticks_per_px)
                    duration_tick = max(int(w * ticks_per_px), 1)  # ≥1 tick

                    events.append((midi_note, start_tick, duration_tick))

                # sort left→right
                return sorted(events, key=lambda t: t[1])


            def write_midi_file(events, tempo_bpm=120, ticks_per_beat=480):
                # 1. Build absolute-time timeline
                timeline = []
                for note, start, dur in events:
                    timeline.append(("on", start, note))
                    timeline.append(("off", start + dur, note))

                timeline.sort(key=lambda t: t[1])  # sort by absolute tick

                # 2. Create file & track
                mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
                track = mido.MidiTrack()
                mid.tracks.append(track)

                tempo = mido.bpm2tempo(tempo_bpm)
                track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

                # 3. Emit messages with non-negative deltas
                prev_tick = 0
                for kind, tick, note in timeline:
                    delta = tick - prev_tick  # guaranteed ≥ 0 after sort
                    msg = "note_on" if kind == "on" else "note_off"
                    track.append(mido.Message(msg, note=note, velocity=64, time=delta))
                    prev_tick = tick

                # Convert to bytes for download
                buf = io.BytesIO()
                mid.save(file=buf)
                buf.seek(0)

                return buf.getvalue()


            # Generate MIDI events
            events = build_note_events(
                st.session_state.quantized_notes,
                st.session_state.lines,
                img_shape[1],
                base_midi_note
            )


            # Create piano roll visualization
            def create_piano_roll(events, img_width, line_count):
                # Determine the time range and note range
                end_time = max(start + dur for _, start, dur in events)
                min_note = min(note for note, _, _ in events)
                max_note = max(note for note, _, _ in events)
                note_range = max_note - min_note + 1

                # Create a blank image for the piano roll
                height = note_range * 10  # 10 pixels per note
                width = int(end_time * 1.1)  # Add some padding
                piano_roll = np.ones((height, width, 3), dtype=np.uint8) * 255

                # Draw horizontal lines for each possible note
                for i in range(note_range + 1):
                    y = i * 10
                    color = (200, 200, 200)  # Light gray
                    if (min_note + i) % 12 in [0, 2, 4, 5, 7, 9, 11]:  # C, D, E, F, G, A, B
                        color = (150, 150, 150)  # Darker gray for white keys
                    cv2.line(piano_roll, (0, y), (width, y), color, 1)

                # Draw the notes as colored rectangles
                for note, start, dur in events:
                    y = (note - min_note) * 10
                    x = start
                    w = dur
                    h = 10

                    # Choose color based on note value (cycle through rainbow)
                    hue = ((note % 12) / 12.0) * 180
                    hsv_color = np.array([[[hue, 200, 200]]], dtype=np.uint8)
                    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0].tolist()

                    cv2.rectangle(piano_roll, (x, height - y - h), (x + w, height - y), rgb_color, -1)
                    cv2.rectangle(piano_roll, (x, height - y - h), (x + w, height - y), (0, 0, 0), 1)

                return piano_roll


            # Generate and display the piano roll
            piano_roll = create_piano_roll(events, img_shape[1], len(st.session_state.lines))
            st.image(piano_roll, caption="Piano Roll Visualization", use_column_width=True)

            # Create MIDI file for download
            midi_data = write_midi_file(events, tempo_bpm=tempo_bpm)

            # Provide download button
            st.download_button(
                label="Download MIDI File",
                data=midi_data,
                file_name="paint_to_music.mid",
                mime="audio/midi"
            )