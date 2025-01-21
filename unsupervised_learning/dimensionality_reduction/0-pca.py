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
    cov = np.cov(X, rowvar=False)

    eig_vals, eig_vecs = np.linalg.eigh(cov)

    idx = np.argsort(eig_vals)[::-1]
    eig_vals_sorted = eig_vals[idx]
    eig_vecs_sorted = eig_vecs[:, idx]

    total_var = np.sum(eig_vals_sorted)
    cumulative_var_ratio = np.cumsum(eig_vals_sorted) / total_var

    num_comp = np.searchsorted(cumulative_var_ratio, var) + 1

    W = eig_vecs_sorted[:, :num_comp]

    return W
