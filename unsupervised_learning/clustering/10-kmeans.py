#!/usr/bin/env python3
"""
Performs K-means clustering on a given dataset using sklearn.cluster.
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        k (int): the number of clusters

    Returns:
        C, clss
            C is a numpy.ndarray of shape (k, d) containing the centroid means
            clss is a numpy.ndarray of shape (n,) containing the index of the
            cluster in C that each data point belongs to
    """
    kmeans_model = sklearn.cluster.KMeans(n_clusters=k)
    kmeans_model.fit(X)

    C = kmeans_model.cluster_centers_
    clss = kmeans_model.labels_
    return C, clss
