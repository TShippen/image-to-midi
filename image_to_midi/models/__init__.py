"""Domain models for the image-to-midi application."""

# Re-export core models
from image_to_midi.models.core_models import NoteBox, MidiEvent

# Re-export pipeline models
from image_to_midi.models.pipeline_models import (
    BinaryResult,
    DetectionResult,
    StaffResult,
    MidiResult,
)

# Re-export setting models
from image_to_midi.models.settings_models import (
    ImageProcessingParams,
    NoteDetectionParams,
    StaffParams,
    MidiParams,
    ProcessingParameters,
)

# Re-export visualization models
from image_to_midi.models.visualization_models import VisualizationSet
