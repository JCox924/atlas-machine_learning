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
