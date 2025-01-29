#!/usr/bin/env python3
"""
Calculates the total intra-cluster variance for a data set.
"""

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set.

    Args:
        X (numpy.ndarray): of shape (n, d) containing the data set
        C (numpy.ndarray): of shape (k, d) containing the centroid means

    Returns:
        var (float): the total variance (sum of squared distances)
        or None on failure
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(C, np.ndarray) or len(C.shape) != 2 or
            X.shape[1] != C.shape[1] or X.size == 0 or C.size == 0):
        return None

    distances = np.linalg.norm(X[:, None] - C, axis=2) ** 2

    var = np.min(distances, axis=1).sum()

    return var
