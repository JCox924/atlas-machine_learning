#!/usr/bin/env python3
"""
Module 13-batch_norm contains functions:
    batch_norm(Z, gamma, beta, epsilon)
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Args:
        Z: a numpy.ndarray of shape (m, n) that should be normalized
            --- where m is the number of examples
            and n is the number of features in Z
        gamma: a numpy.ndarray of shape (1, n)
            containing the scales used for batch normalization
        beta: a numpy.ndarray of shape (1, n)
            containing the offsets used for batch normalization
        epsilon: a small positive number to avoid division by zero
    Returns:
        the normalized Z matrix
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    Z_scaled = gamma * Z_norm + beta

    return Z_scaled
