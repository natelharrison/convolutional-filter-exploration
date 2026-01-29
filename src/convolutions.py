import numpy as np
from typing import Literal

def naive_2d_convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Simple loop-based implementation for educational purposes.
    :param image:
    :param kernel:
    :return:
    """

    if image.ndim != 2:
        raise ValueError('Image must be 2-dimensional')
    if kernel.ndim != 2:
        raise ValueError('Kernel must be 2-dimensional')


    kH, kW = kernel.shape
    if kH % 2 == 0 or kW % 2 == 0:
        raise ValueError('Kernel must be odd')

    pad_h, pad_w = kH // 2, kW // 2
    image_padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    H, W = image.shape

    image_convolution = np.zeros_like(image, dtype=np.float64)
    for y in range(H):
        for x in range(W):
            new_pixel_value = 0
            for k_y in range(kH):
                for k_x in range(kW):
                    new_pixel_value += image_padded[y + k_y, x + k_x] * kernel[k_y, k_x]
            image_convolution[y, x] = new_pixel_value

    return image_convolution.astype(image.dtype)


def numpy_2d_convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Optimized(?) NumPy Implementation.
    :param image:
    :param kernel:
    :return:
    """

    if image.ndim != 2:
        raise ValueError('Image must be 2-dimensional')
    if kernel.ndim != 2:
        raise ValueError('Kernel must be 2-dimensional')

    H, W = image.shape
    kH, kW = kernel.shape
    if kH % 2 == 0 or kW % 2 == 0:
        raise ValueError('Kernel must be odd')

    padding_h, padding_w = kH // 2, kW // 2
    image_padded = np.pad(image, ((padding_h, padding_h), (padding_w, padding_w)), mode='edge')

    s0, s1 = image_padded.strides
    image_views = np.lib.stride_tricks.as_strided(image_padded, shape=(H, W, kH, kW), strides=(s0, s1, s0, s1), writeable=False)

    image_convolution = np.einsum('ijab, ab->ij', image_views, kernel, optimize=True)
    return image_convolution.astype(image.dtype)


def convolve_1d(image:np.ndarray, kernel:np.ndarray, axis:int, pad_mode:Literal['reflect']='reflect')->np.ndarray:
    signal = np.moveaxis(image, axis, -1)

    kernel_size = kernel.shape[0]
    radius = kernel_size // 2

    pad_width = [(0,0)] * signal.ndim
    pad_width[-1] = (radius, radius)
    padded_signal = np.pad(signal, pad_width, mode=pad_mode)

    strides = padded_signal.strides
    *base_shape, axis_length = signal.shape

    windows = np.lib.stride_tricks.as_strided(
        padded_signal,
        (*base_shape, axis_length, kernel_size),
        (*strides[:-1], strides[-1], strides[-1]),
        writeable=False
    )

    # Not sure if I need to implement clipping to dtype min, max
    # That would be a lot of work... (I should be but let's ignore it for now)
    convolved = np.einsum('...lk,k->...l', windows, kernel)
    return np.moveaxis(convolved, -1, axis)


def sequential_convolve(image:np.ndarray, kernel:np.ndarray)->np.ndarray:
    # Ideally I work with float64 and then would clip the output to image.dtype min/max and then cast to image.dtype
    # But that seems tedious for this learning project so we return float32 result
    working_image = image.astype(np.float32, copy=True)
    kernel = kernel.astype(np.float32)
    for ax in range(working_image.ndim):
        working_image = convolve_1d(working_image, kernel, ax)
    return working_image
