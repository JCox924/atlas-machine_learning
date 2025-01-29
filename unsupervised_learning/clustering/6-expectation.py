#!/usr/bin/env python3
"""
Calculates the expectation step in the EM algorithm for a GMM.
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM.

    Args:
        X (numpy.ndarray): of shape (n, d) containing the data set
        pi (numpy.ndarray): of shape (k,) containing the priors for each cluster
        m (numpy.ndarray): of shape (k, d) containing the centroid means for each cluster
        S (numpy.ndarray): of shape (k, d, d) containing the covariance matrices for each cluster

    Returns:
        g, l or None, None on failure
        - g is a numpy.ndarray of shape (k, n) containing the posterior
          probabilities for each data point in each cluster
        - l is the total log likelihood
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0 or
            not isinstance(pi, np.ndarray) or pi.ndim != 1 or pi.size == 0 or
            not isinstance(m, np.ndarray) or m.ndim != 2 or m.size == 0 or
            not isinstance(S, np.ndarray) or S.ndim != 3 or S.size == 0):
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if (m.shape[0] != k or m.shape[1] != d or
            S.shape[0] != k or S.shape[1] != d or S.shape[2] != d):
        return None, None

    if not np.isclose([np.sum(pi)], [1])[0] or (pi < 0).any():
        return None, None

    prob = np.zeros((k, n))

    for j in range(k):
        pdf_vals = pdf(X, m[j], S[j])
        if pdf_vals is None:
            return None, None
        prob[j] = pi[j] * pdf_vals

    total = np.sum(prob, axis=0)

    if np.any(total == 0):
        return None, None

    g = prob / total

    log_likelihood = np.sum(np.log(total))

    return g, log_likelihood
