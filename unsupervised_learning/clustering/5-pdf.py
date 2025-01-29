#!/usr/bin/env python3
"""
Calculates the probability density function of a Gaussian distribution.
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.

    Args:
        X (numpy.ndarray): shape (n, d) with data points
        m (numpy.ndarray): shape (d,) mean of the distribution
        S (numpy.ndarray): shape (d, d) covariance matrix of the distribution

    Returns:
        P (numpy.ndarray): shape (n,) of PDF values for each data point
                           minimum value of 1e-300
                           or None on failure
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(m, np.ndarray) or m.ndim != 1 or
            not isinstance(S, np.ndarray) or S.ndim != 2):
        return None
    n, d = X.shape
    if (m.shape[0] != d or S.shape[0] != d or S.shape[1] != d):
        return None

    try:
        det = np.linalg.det(S)
        if det <= 0:
            return None
        inv_S = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return None

    denom = np.sqrt(((2 * np.pi) ** d) * det)

    diff = X - m

    exponent = -0.5 * np.sum(diff * (diff @ inv_S), axis=1)

    P = (1.0 / denom) * np.exp(exponent)

    P = np.maximum(P, 1e-300)

    return P
