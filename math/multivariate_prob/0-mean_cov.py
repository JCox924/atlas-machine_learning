#!/usr/bin/env python3
"""
Module 0-mean_cov
Contains a function that calculates the mean and covariance of a data set.
"""

import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Parameters
    ----------
    X : numpy.ndarray
        The data set of shape (n, d) where
          n is the number of data points
          d is the number of dimensions in each data point

    Returns
    -------
    mean : numpy.ndarray
        A 1 x d array containing the mean of the data set
    cov : numpy.ndarray
        A d x d array containing the covariance matrix of the data set

    Raises
    ------
    TypeError
        If X is not a 2D numpy.ndarray
    ValueError
        If n < 2
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a 2D numpy.ndarray")
    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)

    X_centered = X - mean
    cov = np.matmul(X_centered.T, X_centered) / (n - 1)

    return mean, cov
