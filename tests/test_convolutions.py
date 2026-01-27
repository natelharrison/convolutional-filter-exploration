import numpy as np
import pytest
import cv2
from scipy.ndimage import convolve as scipy_convolve


from src.convolutions import naive_2d_convolve, numpy_2d_convolve

@pytest.fixture
def random_image():
    np.random.seed(42)
    return np.random.rand(50, 50).astype(np.uint8)

@pytest.fixture
def gaussian_kernel():
    k_size = 5
    k_sigma = 5.0
    k_1d = cv2.getGaussianKernel(k_size, k_sigma)
    k_2d = k_1d @ k_1d.T
    return k_2d

def test_input_validation():
    img = np.zeros((10, 10))
    even_kernel = np.zeros((2, 2))

    with pytest.raises(ValueError):
        naive_2d_convolve(img, even_kernel)


def test_identity_kernel(random_image):
    kernel = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]).reshape(3, 3)
    np.testing.assert_allclose(random_image, naive_2d_convolve(random_image, kernel), rtol=1e-07)


def test_flat_kernel_vs_scipy(random_image):
    kernel = np.full((5, 5), 5)
    kernel //= 25
    np.testing.assert_allclose(
        scipy_convolve(random_image, kernel, mode='nearest'),
        naive_2d_convolve(random_image, kernel),
        rtol=1e-07
    )

def test_naive_vs_numpy(random_image, gaussian_kernel):
    naive_2d_convolution = naive_2d_convolve(random_image, gaussian_kernel)
    numpy_2d_convolution = numpy_2d_convolve(random_image, gaussian_kernel)
    np.testing.assert_allclose(
        naive_2d_convolution,
        numpy_2d_convolution,
        rtol=1e-07
    )