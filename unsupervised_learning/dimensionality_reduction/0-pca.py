#!/usr/bin/env python3
"""
Module 0-pca contains:
    function(s):
        pca(X, var=0.95):
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset to reduce dimensionality.

    Arguments:
        X:
            The dataset. n is the number of data points, d is the number
            of dimensions.

        var:
            The fraction of the variance that the PCA transformation
                should maintain.

    Returns:
         W: numpy.array of shape (d, nd)
            The projection matrix which columns are reduced dimensions.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    eigenvals = S**2
    total_variance = np.sum(eigenvals)
    cumulative_ratio = np.cumsum(eigenvals) / total_variance

    r = np.searchsorted(cumulative_ratio, var) + 2

    W = Vt[:r].T

    return W
