#!/usr/bin/env python3
import numpy as np
"""Shuffle data module contains functions: shuffle_data(X, Y)"""


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Parameters:
    - X: First matrix of shape (m, nx) to shuffle.
    - Y: Second matrix of shape (m, ny) to shuffle.

    Returns:
    - X_shuffled, Y_shuffled: the shuffled X and Y matrices.
    """
    permutation = np.random.permutation(X.shape[0])
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]
    return X_shuffled, Y_shuffled
