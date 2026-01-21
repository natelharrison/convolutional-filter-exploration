import numpy as np

def naive_2d_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
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


    k_h, k_w = kernel.shape
    if k_h % 2 == 0 or k_w % 2 == 0:
        raise ValueError('Kernel must be odd')

    pad_h, pad_w = k_h // 2, k_w // 2
    image_padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    img_h, img_w = image.shape

    output = np.zeros_like(image)
    for y in range(img_h):
        for x in range(img_w):
            new_pixel_value = 0
            for k_y in range(k_h):
                for k_x in range(k_w):
                    new_pixel_value += image_padded[y + k_y, x + k_x] * kernel[k_y, k_x]
            output[y, x] = new_pixel_value

    return output
