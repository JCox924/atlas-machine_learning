#!/usr/bin/env python3
"""
Module 1-regular contains
    function regular(P)
"""
import numpy as np


def regular(P) -> np.ndarray:
    """
    Determines the steady state probabilities of a regular markov chaina

    Args:
        P (numpy.ndarray): shape (n, n) representing the transition matrix
    Returns:
         shape (1, n) containing the steady state probabilities,
            or None on failure

    """
    eigvals, eigvecs = np.linalg.eig(P.T)
    ones = np.isclose(eigvals, 1, atol=1e-8)
    if np.sum(ones) != 1:
        return None

    stationary = eigvecs[:, ones].flatten()
    stationary = np.real_if_close(stationary, tol=1e-8)
    stationary = stationary.astype(float)

    if np.mean(stationary) < 0:
        stationary = -stationary

    total = np.sum(stationary)
    if np.isclose(total, 0, atol=1e-8):
        return None

    stationary = stationary / total

    stationary = stationary.reshape(1, -1)
    return stationary
