#!/usr/bin/env python3
"""
Performs K-means on a dataset.
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Arguments:
        X (numpy.ndarray): of shape (n, d) containing the dataset.
            n is the number of data points
            d is the number of dimensions for each data point
        k (int): positive integer containing the number of clusters
        iterations (int): positive integer containing the maximum number of
            iterations the algorithm should run

    Returns:
        C, clss, or (None, None) on failure
            C is a numpy.ndarray of shape (k, d) with the centroid means
            clss is a numpy.ndarray of shape (n,) with the index of the cluster
            each data point belongs to
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0):
        return None, None

    n, d = X.shape
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    C = np.random.uniform(mins, maxs, size=(k, d))

    used_reinit = False

    for _ in range(iterations):

        distances = np.linalg.norm(X[:, None] - C, axis=2)

        clss = np.argmin(distances, axis=1)

        old_C = C.copy()

        new_C = np.zeros_like(C)

        empty_clusters = []
        for cluster_idx in range(k):
            points_in_cluster = X[clss == cluster_idx]
            if len(points_in_cluster) == 0:
                empty_clusters.append(cluster_idx)
            else:
                new_C[cluster_idx] = points_in_cluster.mean(axis=0)

        if empty_clusters and not used_reinit:
            reinit_vals = np.random.uniform(mins, maxs,
                                            size=(len(empty_clusters), d))
            used_reinit = True
            for i, cluster_idx in enumerate(empty_clusters):
                new_C[cluster_idx] = reinit_vals[i]
        else:
            for cluster_idx in empty_clusters:
                new_C[cluster_idx] = old_C[cluster_idx]

        if np.allclose(old_C, new_C):
            C = new_C
            break

        C = new_C

    distances = np.linalg.norm(X[:, None] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
