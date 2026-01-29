import numpy as np

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

