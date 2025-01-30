#!/usr/bin/env python3
"""
Performs the expectation-maximization algorithm for a GMM.
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation-maximization for a GMM.

    Args:
        X (numpy.ndarray): shape (n, d) containing the data set
        k (int): positive integer, number of clusters
        iterations (int): maximum number of iterations for the algorithm
        tol (float): non-negative tolerance for early stopping
        verbose (bool): if True, print progress every 10 iterations and
                        after the last iteration

    Returns:
        pi, m, S, g, l, or (None, None, None, None, None) on failure
            pi is a numpy.ndarray of shape (k,) with updated priors
            m is a numpy.ndarray of shape (k, d) with updated means
            S is a numpy.ndarray of shape (k, d, d) with updated covariances
            g is a numpy.ndarray of shape (k, n) with posterior probabilities
            l is the log likelihood of the model
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    log_l_old = 0
    for i in range(iterations):
        g, log_l = expectation(X, pi, m, S)
        if g is None or log_l is None:
            return None, None, None, None, None

        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0):
            print(f"Log Likelihood after {i} iterations: {log_l:.5f}")

        if i > 0:
            if abs(log_l - log_l_old) <= tol:
                break

        log_l_old = log_l

    if verbose:
        print(f"Log Likelihood after {i} iterations: {log_l:.5f}")

    return pi, m, S, g, log_l
