"""
Pipeline processing functions for image-to-MIDI conversion.

This module contains the core processing functions for the image-to-MIDI pipeline,
separating business logic from visualization and UI concerns.
"""

import tempfile
import logging
import cv2

from image_to_midi.image_processing import preprocess_image, create_note_visualization
from image_to_midi.note_detection import detect_notes as detect_note_boxes_raw
from image_to_midi.staff import (
    detect_lines,
    average_box_height,
    adjust_box_height,
    vertical_quantize_notes,
    calculate_fit_accuracy,
    calculate_note_variation,
    create_staff_visualization,
)
from image_to_midi.midi_utils import (
    build_note_events,
    write_midi_file,
    create_piano_roll,
)

from image_to_midi.models.core_models import NoteBox
from image_to_midi.models.pipeline_models import (
    BinaryResult,
    DetectionResult,
    StaffResult,
    MidiResult,
)

from image_to_midi.models.visualization_models import VisualizationSet

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Custom exceptions
class PipelineError(Exception):
    """Base exception for pipeline processing errors."""

    pass


class InputError(PipelineError):
    """Exception raised when input data is invalid."""

    pass


class ProcessingError(PipelineError):
    """Exception raised when processing fails."""

    pass


# Core Processing Functions - Each does one thing


def process_binary_image(image, params):
    """Convert image to binary mask.

    Args:
        image: RGB image as a NumPy array
        params: Image processing parameters

    Returns:
        BinaryResult containing the binary mask
    """
    try:
        if image is None:
            logger.warning("No image provided for processing")
            return BinaryResult()

        # Convert from RGB to BGR for OpenCV
        bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Process image
        binary_mask = preprocess_image(bgr_img, params.threshold)

        return BinaryResult(binary_mask=binary_mask)
    except Exception as e:
        logger.error(f"Error in binary image processing: {str(e)}")
        return BinaryResult()


def detect_notes(binary_result, params):
    """Detect note-like shapes in the binary image.

    Args:
        binary_result: Binary mask from preprocessing
        params: Note detection parameters

    Returns:
        DetectionResult containing detected note boxes
    """
    try:
        binary_mask = binary_result.binary_mask
        if binary_mask is None:
            logger.warning("No binary mask provided for note detection")
            return DetectionResult()

        # Detect notes
        raw_boxes = detect_note_boxes_raw(
            binary_mask,
            params.min_area,
            params.max_area,
            params.min_aspect_ratio,
            params.max_aspect_ratio,
        )

        # Convert to NoteBox models
        note_boxes = [
            NoteBox(x=x, y=float(y), w=w, h=float(h)) for x, y, w, h in raw_boxes
        ]

        return DetectionResult(note_boxes=note_boxes)
    except Exception as e:
        logger.error(f"Error in note detection: {str(e)}")
        return DetectionResult()


def create_staff(detection_result, params):
    """Create staff lines and quantize notes.

    Args:
        detection_result: Detection result with note boxes
        params: Staff creation parameters

    Returns:
        StaffResult with staff lines and quantized notes
    """
    try:
        note_boxes = detection_result.note_boxes
        if not note_boxes:
            logger.warning("No note boxes available for staff creation")
            return StaffResult()

        # Choose working list of NoteBoxes based on method
        if params.method == "Original":
            working_boxes = note_boxes
        elif params.method == "Average":
            working_boxes = average_box_height(note_boxes)
        else:  # Adjustable
            working_boxes = adjust_box_height(note_boxes, params.height_factor)

        # Create staff lines
        lines = detect_lines(working_boxes, params.num_lines)

        # Quantize notes to staff lines
        quantized = vertical_quantize_notes(working_boxes, lines)

        # Calculate metrics
        fit_accuracy = calculate_fit_accuracy(working_boxes, lines)
        pitch_variation = calculate_note_variation(working_boxes, lines)

        return StaffResult(
            lines=lines,
            original_boxes=working_boxes,
            quantized_boxes=quantized,
            fit_accuracy=fit_accuracy,
            pitch_variation=pitch_variation,
        )
    except Exception as e:
        logger.error(f"Error in staff creation: {str(e)}")
        return StaffResult()


def generate_midi(staff_result, params):
    """Generate MIDI from staff notes.

    Args:
        staff_result: Staff and quantized notes
        params: MIDI generation parameters

    Returns:
        MidiResult with MIDI events and data
    """
    try:
        if not staff_result.quantized_boxes:
            logger.warning("No quantized note boxes for MIDI generation")
            return MidiResult()

        if staff_result.lines.size == 0:
            logger.warning("No staff lines for MIDI generation")
            return MidiResult()

        # Build MIDI events
        events = build_note_events(
            staff_result.quantized_boxes, staff_result.lines, params.base_midi_note
        )

        # Generate MIDI file
        midi_bytes = write_midi_file(events, params.tempo_bpm)

        # Save MIDI to a temporary file for playback
        midi_file_path = ""
        if midi_bytes:
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
                tmp.write(midi_bytes)
                midi_file_path = tmp.name

        # Get base note name
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        note = note_names[params.base_midi_note % 12]
        octave = (params.base_midi_note // 12) - 1
        base_note_name = f"{note}{octave}"

        return MidiResult(
            events=events,
            midi_bytes=midi_bytes,
            midi_file_path=midi_file_path,
            base_note_name=base_note_name,
        )
    except Exception as e:
        logger.error(f"Error in MIDI generation: {str(e)}")
        return MidiResult()


# Visualization Functions - Each returns a specific visualization


def create_detection_rgb_visualization(image, binary_result, detection_result):
    """Create RGB visualization of detected notes.

    Args:
        image: Original RGB image
        binary_result: Binary processing result
        detection_result: Detection results with note boxes

    Returns:
        RGB visualization with boxes drawn on original image or None if error occurs
    """
    try:
        if image is None:
            logger.warning("No image provided for RGB detection visualization")
            return None

        if binary_result.binary_mask is None:
            logger.warning("No binary mask for RGB detection visualization")
            return None

        if not detection_result.note_boxes:
            logger.warning("No note boxes for RGB detection visualization")
            return None

        # Convert to BGR for processing
        bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Get raw box tuples for visualization
        raw_boxes = [
            (box.x, int(box.y), box.w, int(box.h))
            for box in detection_result.note_boxes
        ]

        # Create visualizations
        vis_rgb, _ = create_note_visualization(
            bgr_img, binary_result.binary_mask, raw_boxes
        )

        # Convert BGR back to RGB for display
        return cv2.cvtColor(vis_rgb, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error in RGB detection visualization: {str(e)}")
        return None


def create_detection_binary_visualization(image, binary_result, detection_result):
    """Create binary visualization of detected notes.

    Args:
        image: Original RGB image
        binary_result: Binary processing result
        detection_result: Detection results with note boxes

    Returns:
        Binary visualization with boxes drawn on binary mask or None if error occurs
    """
    try:
        if image is None:
            logger.warning("No image provided for binary detection visualization")
            return None

        if binary_result.binary_mask is None:
            logger.warning("No binary mask for binary detection visualization")
            return None

        if not detection_result.note_boxes:
            logger.warning("No note boxes for binary detection visualization")
            return None

        # Convert to BGR for processing
        bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Get raw box tuples for visualization
        raw_boxes = [
            (box.x, int(box.y), box.w, int(box.h))
            for box in detection_result.note_boxes
        ]

        # Create visualizations
        _, vis_bin = create_note_visualization(
            bgr_img, binary_result.binary_mask, raw_boxes
        )

        return vis_bin
    except Exception as e:
        logger.error(f"Error in binary detection visualization: {str(e)}")
        return None


def create_staff_original_visualization(image_shape, staff_result):
    """Create visualization for original staff lines and notes.

    Args:
        image_shape: Original image dimensions (height, width)
        staff_result: Staff results with lines and notes

    Returns:
        Visualization of original notes with staff lines or None if error occurs
    """
    try:
        if not staff_result.original_boxes:
            logger.warning("No original boxes for staff visualization")
            return None

        if staff_result.lines.size == 0:
            logger.warning("No staff lines for original staff visualization")
            return None

        # Create visualization
        return create_staff_visualization(
            image_shape, staff_result.original_boxes, staff_result.lines
        )
    except Exception as e:
        logger.error(f"Error in original staff visualization: {str(e)}")
        return None


def create_staff_quantized_visualization(image_shape, staff_result):
    """Create visualization for quantized notes on staff lines.

    Args:
        image_shape: Original image dimensions (height, width)
        staff_result: Staff results with lines and notes

    Returns:
        Visualization of quantized notes with staff lines or None if error occurs
    """
    try:
        if not staff_result.quantized_boxes:
            logger.warning("No quantized boxes for staff visualization")
            return None

        if staff_result.lines.size == 0:
            logger.warning("No staff lines for quantized staff visualization")
            return None

        # Create visualization
        return create_staff_visualization(
            image_shape, staff_result.quantized_boxes, staff_result.lines
        )
    except Exception as e:
        logger.error(f"Error in quantized staff visualization: {str(e)}")
        return None


def create_piano_roll_visualization(midi_result):
    """Create piano roll visualization for MIDI events.

    Args:
        midi_result: MIDI generation result with events

    Returns:
        Piano roll image as numpy array or None if error occurs
    """
    try:
        if not midi_result.events:
            logger.warning("No MIDI events for piano roll visualization")
            return None

        # Create piano roll
        return create_piano_roll(midi_result.events)
    except Exception as e:
        logger.error(f"Error in piano roll visualization: {str(e)}")
        return None


def create_visualization_set(
    image, binary_result, detection_result, staff_result, midi_result
):
    """Create a complete set of visualizations for the pipeline.

    Args:
        image: Original RGB image
        binary_result: Binary processing result
        detection_result: Note detection result
        staff_result: Staff creation result
        midi_result: MIDI generation result

    Returns:
        Complete set of visualizations for UI
    """
    try:
        if image is None:
            logger.warning("No image provided for visualization set")
            return VisualizationSet()

        # Generate all visualizations
        note_vis_rgb = create_detection_rgb_visualization(
            image, binary_result, detection_result
        )

        note_vis_bin = create_detection_binary_visualization(
            image, binary_result, detection_result
        )

        img_shape = image.shape[:2]
        staff_vis = create_staff_original_visualization(img_shape, staff_result)
        quant_vis = create_staff_quantized_visualization(img_shape, staff_result)

        piano_roll = create_piano_roll_visualization(midi_result)

        # Package into visualization set
        return VisualizationSet(
            binary_mask=binary_result.binary_mask,
            note_detection=note_vis_rgb,
            note_detection_binary=note_vis_bin,
            staff_lines=staff_vis,
            quantized_notes=quant_vis,
            piano_roll=piano_roll,
        )
    except Exception as e:
        logger.error(f"Error in visualization set creation: {str(e)}")
        return VisualizationSet()


def process_complete_pipeline(
    image, image_params, detection_params, staff_params, midi_params
):
    """Process the complete image-to-MIDI pipeline.

    Args:
        image: Input RGB image
        image_params: Image processing parameters
        detection_params: Note detection parameters
        staff_params: Staff creation parameters
        midi_params: MIDI generation parameters

    Returns:
        Tuple of (binary_result, detection_result, staff_result, midi_result, visualizations)
    """
    try:
        if image is None:
            logger.warning("No image provided for pipeline processing")
            return (
                BinaryResult(),
                DetectionResult(),
                StaffResult(),
                MidiResult(),
                VisualizationSet(),
            )

        # Step 1: Image processing
        binary_result = process_binary_image(image, image_params)

        # Step 2: Note detection
        detection_result = detect_notes(binary_result, detection_params)

        # Step 3: Staff creation
        staff_result = create_staff(detection_result, staff_params)

        # Step 4: MIDI generation
        midi_result = generate_midi(staff_result, midi_params)

        # Create visualizations as a separate step
        vis_set = create_visualization_set(
            image, binary_result, detection_result, staff_result, midi_result
        )

        return binary_result, detection_result, staff_result, midi_result, vis_set

    except Exception as e:
        logger.error(f"Error in pipeline processing: {str(e)}")
        # Return empty results on error
        return (
            BinaryResult(),
            DetectionResult(),
            StaffResult(),
            MidiResult(),
            VisualizationSet(),
        )
