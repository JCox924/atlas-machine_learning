#!/usr/bin/env python3
"""
Module 1-pca contains:
    function(s):
        pca(X, ndim):
"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset to reduce dimensionality.

    Arguments:
        X:
            The dataset. n is the number of data points, d is the number
            of dimensions of each point.

        ndim:
            The new dimensionality of the transformed X.

    Returns:
         T: numpy.array of shape (n, ndim)
            The projection matrix containing the transformed version of X.
    """
    X_centered = X - X.mean(axis=0)

    U, s, Vh = np.linalg.svd(X_centered, full_matrices=False)

    W = Vh[:ndim].T

    T = np.dot(X_centered, W)

    return T
