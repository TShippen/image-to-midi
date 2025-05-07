import numpy as np
from src.image_processing import preprocess_image, create_note_visualization


def test_preprocess_image():
    img = np.full((20, 20, 3), 200, dtype=np.uint8)
    binary = preprocess_image(img, 100)
    assert binary.shape == (20, 20)
    assert binary.dtype == np.uint8


def test_create_note_visualization():
    orig = np.zeros((10, 10, 3), dtype=np.uint8)
    binary = np.zeros((10, 10), dtype=np.uint8)
    notes = [(1, 1, 3, 3)]
    rgb, bin_rgb = create_note_visualization(orig, binary, notes)
    assert rgb.shape == (10, 10, 3)
    assert bin_rgb.shape == (10, 10, 3)
