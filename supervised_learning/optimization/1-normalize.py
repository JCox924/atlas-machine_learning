#!/usr/bin/env python3
import numpy as np
"""Normalize Module"""

def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Parameters:
    - X (numpy.ndarray): Matrix of shape (d, nx) to normalize, where
      d is the number of data points and nx is the number of features.
    - m (numpy.ndarray): Mean of all features of X, of shape (nx,).
    - s (numpy.ndarray): Standard deviation of all features of X, of shape (nx,).

    Returns:
    - X_normalized (numpy.ndarray): The normalized X matrix.
    """
    X_normalized = (X - m) / s
    return X_normalized