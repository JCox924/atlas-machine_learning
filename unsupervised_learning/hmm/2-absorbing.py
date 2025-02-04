#!/usr/bin/env python3
"""
Module 2-absorbing contains
    function:
        absorbing(P)
"""
import numpy as np


def absorbing(P)-> bool:
    """
    Args:
        P(np.ndarray):  square 2D array of shape (n, n) representing
         the standard transition matrix
    Returns:
        Returns: True if it is absorbing, or False on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False
    if not np.allclose(P.sum(axis=1), 1, atol=1e-8) or np.any(P < 0) or not np.all(np.isfinite(P)):
        return False

    n =P.shape[0]

    absoring_states = []

    for i in range(n):
        row_no_i = np.concatenate((P[i, :i], P[i, i+1:]))
        if np.isclose(P[i, i], 1, atol=1e-8) and np.allclose(row_no_i, 0, atol=1e-8):
            absoring_states.append(i)
        if len(absoring_states) == 0:
            return False


    def can_reach(i, v):
        if i in absoring_states:
            return True
        for j in range(n):
            if P[i, j] > 0 and j not in v:
                v.add(j)
                if can_reach(j, v):
                    return True
        return False

    for i in range(n):
        if not can_reach(i, set([i])):
            return False
    return True
