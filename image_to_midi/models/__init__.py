"""Domain models for the image-to-midi application.

This module provides a centralized location for all data models used throughout
the image-to-MIDI conversion pipeline. It includes:

- Core domain models (NoteBox, MidiEvent)
- Pipeline processing stage results (BinaryResult, DetectionResult, etc.)
- Configuration parameters for each processing stage
- Visualization data containers

All models are built using Pydantic for data validation and serialization,
ensuring type safety and clear interfaces between pipeline components.
"""

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
