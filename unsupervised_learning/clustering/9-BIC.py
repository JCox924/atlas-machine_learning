#!/usr/bin/env python3
"""
Finds the best number of clusters for a GMM using the Bayesian Information Criterion.
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian Information Criterion (BIC).

    Args:
        X (numpy.ndarray): shape (n, d) containing the data set
        kmin (int): positive integer, minimum number of clusters (inclusive)
        kmax (int): positive integer, maximum number of clusters (inclusive); if None,
                    it is set to the maximum number of data points (n)
        iterations (int): maximum number of iterations for the EM algorithm
        tol (float): non-negative float for tolerance in the EM algorithm
        verbose (bool): if True, print EM progress

    Returns:
        best_k, best_result, l, b or None, None, None, None on failure
            best_k is the best value for k based on its BIC
            best_result is the tuple (pi, m, S) for the best number of clusters
            l is a numpy.ndarray of shape (kmax - kmin + 1) of log-likelihoods
            b is a numpy.ndarray of shape (kmax - kmin + 1) of BIC values
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0 or
            not isinstance(kmin, int) or kmin < 1 or
            (kmax is not None and (not isinstance(kmax, int) or kmax < 1)) or
            not isinstance(iterations, int) or iterations < 1 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if kmax < kmin:
        return None, None, None, None

    ks = np.arange(kmin, kmax + 1)
    l_vals = np.zeros_like(ks, dtype=float)
    b_vals = np.zeros_like(ks, dtype=float)

    results = [None] * ks.size

    for i, k_ in enumerate(ks):
        pi, m, S, g, log_l = expectation_maximization(X, k_, iterations, tol, verbose)

        if (pi is None or m is None or S is None or g is None or log_l is None):
            return None, None, None, None

        l_vals[i] = log_l
        results[i] = (pi, m, S)

        p = (k_ - 1) + (k_ * d) + (k_ * (d * (d + 1) // 2))

        b_vals[i] = p * np.log(n) - 2 * log_l

    best_idx = np.argmin(b_vals)
    best_k = ks[best_idx]
    best_result = results[best_idx]

    return best_k, best_result, l_vals, b_vals
