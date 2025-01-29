#!/usr/bin/env python3
"""
Determines the optimum number of clusters by variance.
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.

    Args:
        X (numpy.ndarray): of shape (n, d) containing the data set
        kmin (int): minimum number of clusters to check for (inclusive)
        kmax (int): maximum number of clusters to check for (inclusive)
        iterations (int): maximum number of iterations for K-means

    Returns:
        results, d_vars, or (None, None) on failure
            - results is a list of (C, clss) from K-means for each cluster size
            - d_vars is a list of the difference in variance from the smallest
              cluster size (kmin) for each cluster size
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(kmin, int) or kmin < 1 or
            (kmax is not None and (not isinstance(kmax, int) or kmax < 1)) or
            not isinstance(iterations, int) or iterations < 1):
        return None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if kmax < kmin or (kmax - kmin) < 1:
        return None, None

    results = []
    variances = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations=iterations)
        if C is None or clss is None:
            return None, None

        var = variance(X, C)
        if var is None:
            return None, None

        results.append((C, clss))
        variances.append(var)

    d_vars = [variances[0] - v for v in variances]

    return results, d_vars
