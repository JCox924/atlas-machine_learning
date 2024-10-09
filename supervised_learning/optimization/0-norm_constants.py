#!/usr/bin/env python3
import numpy as np
""" Normalization Constansts Module"""


def normalization_constants(X):
    """
    Calculates the normalization constants of a matrix.

    Parameters:
    - X: matrix of shape (m, nx) to normalize, where
      m is the number of data points and nx is the number of features.

    Returns:
    - mean: Mean of each feature.
    - std: Standard deviation of each feature.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
