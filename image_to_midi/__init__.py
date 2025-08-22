"""Image-to-MIDI conversion library.

This package provides a comprehensive pipeline for converting images containing
musical notation or paint splatters into MIDI files. It includes image processing,
note detection, staff line generation, and MIDI synthesis capabilities.

The main processing pipeline consists of:
1. Image preprocessing and binarization
2. Note detection using contour analysis
3. Staff line detection and note quantization
4. MIDI event generation and file export
5. Visualization of all processing stages

Example:
    Basic usage through the pipeline API:
    
    >>> from image_to_midi.pipeline import process_complete_pipeline
    >>> from image_to_midi.models import ProcessingParameters
    >>> 
    >>> # Load image and process with default parameters
    >>> image = cv2.imread("musical_image.jpg")
    >>> params = ProcessingParameters()
    >>> results = process_complete_pipeline(image, *params)
"""
