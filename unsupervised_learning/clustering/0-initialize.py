#!/usr/bin/env python3
"""
Module that initializes cluster centroids for K-means.
"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means using a uniform distribution.

    Arguments:
        X (numpy.ndarray): of shape (n, d) containing the dataset to be used
            for K-means clustering
        k (int): the number of clusters

    Returns:
        numpy.ndarray of shape (k, d) containing the initialized centroids,
        or None on failure
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0):
        return None

    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    centroids = np.random.uniform(mins, maxs, (k, X.shape[1]))

    return centroids
