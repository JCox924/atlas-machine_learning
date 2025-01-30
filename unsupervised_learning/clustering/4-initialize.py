#!/usr/bin/env python3
"""
Initializes variables for a Gaussian Mixture Model.
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.

    Arguments:
        X (numpy.ndarray): of shape (n, d), the data set
        k (int): positive integer for the number of cluster

    Returns:
        pi, m, S or None, None, None on failure
            pi: shape (k,) containing the priors
            m: shape (k, d) centroid means for each cluster
            S: shape (k, d, d) covariance matrices
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(k, int) or k <= 0 or
            X.shape[0] < k):
        return None, None, None

    C, clss = kmeans(X, k)
    if C is None or clss is None:
        return None, None, None

    n, d = X.shape

    pi = np.full(shape=(k,), fill_value=1/k)

    m = C

    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
