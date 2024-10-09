#!/usr/bin/env python3
"""Module contains function create_mini_batches(X, Y, batch_size)"""


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches to be used for training a
    neural network using mini-batch gradient descent.

    Args:
        X: Input data of shape, where m is the number of
        data points and nx is the number of features.
        Y: Labels of shape, where m is the number of
        data points and ny is the number of classes for classification tasks.
        batch_size: Number of data points in a batch.

    Returns:
        - mini_batches: List of mini-batches,
        each containing a tuple (X_batch, Y_batch).
    """
    shuffle_data = __import__('2-shuffle_data').shuffle_data
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = []

    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
