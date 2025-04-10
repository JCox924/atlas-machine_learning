#!/usr/bin/env python3
"""
Module 0-likelihood
Contains a function that calculates the likelihood of obtaining data from a
binomial distribution given various hypothetical probabilities.
"""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining the data x and n for each probability
    in the array P, using the binomial distribution.

    Parameters
    ----------
    x : int
        The number of patients that develop severe side effects.
    n : int
        The total number of patients observed.
    P : numpy.ndarray
        A 1D array of various hypothetical probabilities of developing
        severe side effects.

    Returns
    -------
    numpy.ndarray
        A 1D array containing the likelihood of obtaining the data (x and n)
        for each probability in P, respectively.

    Raises
    ------
    ValueError
        - If n is not a positive integer.
        - If x is not an integer >= 0.
        - If x > n.
    TypeError
        - If P is not a 1D numpy.ndarray.
    ValueError
        - If any value in P is out of the range [0, 1].
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is"
                         " greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    def factorial(k):
        if k < 2:
            return 1
        f = 1
        for i in range(2, k + 1):
            f *= i
        return f

    binom_coeff = factorial(n) / (factorial(x) * factorial(n - x))

    L = binom_coeff * (P ** x) * ((1 - P) ** (n - x))

    return L
