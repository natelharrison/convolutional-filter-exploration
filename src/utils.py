from time import perf_counter
from typing import Tuple, Callable

import cv2
import numpy as np

def gaussian_2d_kernel(size:int = 5, sigma:float = 5.0) -> np.ndarray:
    k_1d = cv2.getGaussianKernel(size, sigma)
    k_2d = k_1d @ k_1d.T
    return k_2d

def run_and_time_function(function:Callable, *args, **kwargs) -> Tuple[float, np.ndarray]:
    time_start = perf_counter()
    output = function(*args, **kwargs)
    time_end = perf_counter()
    time_delta = time_end - time_start
    return time_delta, output
