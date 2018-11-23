import numpy as np


def normalize_shape(shape: np.ndarray,
              size: float):
    normalized_shape = (shape - size/2)/(size/2)
    return normalized_shape


def denormalize_shape(norm_shape: np.ndarray,
                      size: float):
    orig_shape = norm_shape*(size/2) + size/2
    return orig_shape
