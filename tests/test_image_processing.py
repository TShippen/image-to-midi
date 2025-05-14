import numpy as np
from image_to_midi.image_processing import preprocess_image


def test_preprocess_image_thresholding(small_rgb_image):
    # Convert our fixture to BGR so grayscale==original
    # Threshold at 100: red(76)>100→0, green(150)>100→0, blue(29)<100→255, black(0)<100→255
    bin_img = preprocess_image(small_rgb_image[..., ::-1], threshold_value=100)
    expected = np.array([[255, 0], [255, 255]], dtype=np.uint8)
    assert np.array_equal(bin_img, expected)


def test_preprocess_image_dtype_and_shape(small_rgb_image):
    b = preprocess_image(small_rgb_image[..., ::-1], threshold_value=50)
    assert b.dtype == np.uint8
    assert b.shape == small_rgb_image.shape[:2]
