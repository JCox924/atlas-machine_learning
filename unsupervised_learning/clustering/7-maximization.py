#!/usr/bin/env python3
"""
Calculates the maximization step in the EM algorithm for a GMM.
"""

import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.

    Args:
        X (numpy.ndarray): shape (n, d) containing the data set
        g (numpy.ndarray): shape (k, n) containing the posterior probabilities
                           for each data point in each cluster

    Returns:
        pi, m, S or None, None, None on failure
            pi is a numpy.ndarray of shape (k,) containing the updated
               priors for each cluster
            m is a numpy.ndarray of shape (k, d) containing the updated
              centroid means for each cluster
            S is a numpy.ndarray of shape (k, d, d) containing the updated
              covariance matrices for each cluster
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0 or
            not isinstance(g, np.ndarray) or g.ndim != 2 or g.size == 0):
        return None, None, None

    n, d = X.shape
    k, n2 = g.shape
    if n2 != n:
        return None, None, None

    if not np.allclose(g.sum(axis=0), np.ones(n)):
        return None, None, None

    sum_g = g.sum(axis=1)

    pi = sum_g / n

    m = (g @ X) / sum_g[:, None]

    S = np.zeros((k, d, d))
    for j in range(k):
        X_centered = X - m[j]

        gamma = g[j][:, np.newaxis]
        cov_j = (X_centered.T @ (gamma * X_centered)) / sum_g[j]
        S[j] = cov_j

    return pi, m, S
