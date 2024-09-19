#!/usr/bin/env python3
"""Module contains function one_hot_encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix
    Y: numpy.ndarray of shape (m,) containing numeric class labels
    classes: maximum number of classes found in Y
    Returns: a one-hot encoding of Y with shape (classes, m), or None on failure
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    if classes <= 0 or np.any(Y >= classes) or np.any(Y < 0):
        return None

    try:
        one_hot = np.zeros((classes, Y.shape[0]))
        one_hot[Y, np.arange(Y.shape[0])] = 1
        return one_hot
    except Exception:
        return None
