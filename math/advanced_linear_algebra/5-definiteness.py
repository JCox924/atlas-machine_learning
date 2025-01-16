#!/usr/bin/env python3
"""
Module definiteness
Contains a function definiteness that classifies the definiteness of a square matrix.
"""

import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a square matrix.

    Parameters
    ----------
    matrix : numpy.ndarray of shape (n, n)
        The matrix whose definiteness should be calculated.

    Returns
    -------
    str or None
        One of:
          - 'Positive definite'
          - 'Positive semi-definite'
          - 'Negative semi-definite'
          - 'Negative definite'
          - 'Indefinite'
        or None if the matrix does not fit any of the above definitions.

    Raises
    ------
    TypeError
        If matrix is not a numpy.ndarray.
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    n = matrix.shape[0]
    if n == 0:
        return None

    if not np.allclose(matrix, matrix.T, atol=1e-8):
        return None  # Not symmetric => can't classify definiteness in the usual sense

    eigenvals = np.linalg.eigvalsh(matrix)

    if np.all(eigenvals > 0):
        return "Positive definite"
    elif np.all(eigenvals >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvals < 0):
        return "Negative definite"
    elif np.all(eigenvals <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
