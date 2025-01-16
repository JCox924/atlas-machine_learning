#!/usr/bin/env python3
"""
Module 1-correlation contains:
    functions:
    correlation(C)
"""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix from a given covariance matrix.

    Arguments:
        C : numpy.ndarray
            A covariance matrix of shape (d, d) where d is the number of dimensions.

    Returns:
        numpy.ndarray
            A correlation matrix of shape (d, d) corresponding to the input covariance matrix.

    Raises:
        TypeError
            If C is not a numpy.ndarray.
        ValueError
            If C does not have shape (d, d).
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std_devs = np.sqrt(np.diag(C))

    # corr(i, j) = C(i, j) / (std_devs[i] * std_devs[j])
    correlation_matrix = C / np.outer(std_devs, std_devs)

    return correlation_matrix
