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
            d is the number of dimensions of each data point
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

    min_v = np.min(X, axis=0)
    max_v = np.max(X, axis=0)

    centroids = np.random.uniform(low=min_v, high=max_v, size=(k, d))

    def assign_labels(data, centers):
        distances = np.linalg.norm(data[:, None] - centers, axis=-1)
        return np.argmin(distances, axis=-1)

    for _ in range(iterations):
        labels = assign_labels(X, centroids)

        new_centroids = np.zeros((k, d))
        for i in range(k):
            points_in_cluster = X[labels == i]
            if len(points_in_cluster) == 0:
                new_centroids[i] = np.random.uniform(low=min_v, high=max_v, size=d)
            else:
                new_centroids[i] = points_in_cluster.mean(axis=0)

        if np.allclose(new_centroids, centroids):
            labels = assign_labels(X, centroids)
            return centroids, labels

        centroids = new_centroids

    labels = assign_labels(X, centroids)
    return centroids, labels
