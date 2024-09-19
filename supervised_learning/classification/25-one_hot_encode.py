#!/usr/bin/env python3
"""module contains function one_hot"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels
    one_hot: numpy.ndarray with shape (classes, m)
    Returns: a numpy.ndarray with shape (m,) containing the numeric labels for each example, or None on failure
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    try:
        labels = np.argmax(one_hot, axis=0)
        return labels
    except Exception:
        return None
