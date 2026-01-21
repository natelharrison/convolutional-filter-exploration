import numpy as np
import pytest
from scipy.ndimage import convolve as scipy_convolve

from src.convolutions import naive_2d_convolution

@pytest.fixture
def random_image():
    np.random.seed(42)
    return np.random.rand(50, 50)


def test_input_validation():
    img = np.zeros((10, 10))
    even_kernel = np.zeros((2, 2))

    with pytest.raises(ValueError):
        naive_2d_convolution(img, even_kernel)


def test_identity_kernel(random_image):
    kernel = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]).reshape(3, 3)
    np.testing.assert_allclose(random_image, naive_2d_convolution(random_image, kernel), rtol=1e-07)


def test_flat_kernel_vs_scipy(random_image):
    kernel = np.full((5, 5), 5)
    kernel //= 25
    np.testing.assert_allclose(
        scipy_convolve(random_image, kernel, mode='nearest'),
        naive_2d_convolution(random_image, kernel),
        rtol=1e-07
    )