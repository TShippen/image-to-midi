# Image to MIDI Converter

A Python application that guides users though the process and choices of creating MIDI from an image using computer vision and musical transformations. The application analyzes visual elements in images (particularly paint splatters or abstract patterns) and transforms them into musical notes, creating unique compositions from visual art.

## Features

- **Visual Note Detection**: Automatically detects visual elements in images and interprets them as musical notes
- **Staff Line Mapping**: Maps detected elements to musical staff lines for pitch determination
- **Musical Transformations**: Apply scales, tempo adjustments, and rhythm quantization
- **Interactive Web Interface**: Real-time parameter adjustment with visual feedback
- **Multiple Output Formats**: Generate both MIDI files and audio (WAV) files

## Requirements

- Python 3.12 or higher
- Poetry (for dependency management) OR pip with venv

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/image-to-midi.git
cd image-to-midi
```

### 2. Choose Your Installation Method

#### Using Poetry

Install dependencies:
```bash
# Install all dependencies (including dev tools like pytest, black)
poetry install

# Or install only production dependencies (no dev tools)
poetry install --no-dev
```

#### Using pip and venv

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -e .
```

Both methods will install all required dependencies including:
- Gradio (web interface)
- OpenCV (image processing)
- NumPy (numerical operations)
- Mido (MIDI file creation)
- Pretty-MIDI (MIDI manipulation)
- Music21 (music theory operations)
- Pydantic (data validation)

## Running the Application

### Start the Gradio Web Interface

#### If you used Poetry:
```bash
poetry run python image_to_midi/gradio_app.py
```

#### If you used pip/venv:
```bash
# Make sure your venv is activated first
python image_to_midi/gradio_app.py
```

The application will start and display:
```
Running on local URL: http://127.0.0.1:7860
```

Open your web browser and navigate to `http://localhost:7860` to use the application.

## Usage

1. **Image Processing**: The application demonstrates the pipeline with a provided paint splatter image. Adjust the threshold slider to control which elements are detected as potential notes.

2. **Note Detection**: Fine-tune the detection parameters:
   - **Min/Max Area**: Control the size range of detected notes
   - **Min/Max Aspect Ratio**: Filter notes by their shape

3. **Staff Line Fitting**: Choose how notes are mapped to pitches:
   - **Original**: Preserves relative heights
   - **Average**: Normalizes all notes to same height
   - **Adjustable**: Custom height scaling

4. **MIDI Generation**: Configure musical output:
   - **Base MIDI Note**: Set the lowest pitch
   - **Tempo**: Control playback speed
   - **Scale Fitting**: Snap notes to musical scales
   - **Rhythm Quantization**: Align notes to a rhythmic grid

5. **Output**: Download the generated MIDI file or listen to the audio preview directly in the browser.
