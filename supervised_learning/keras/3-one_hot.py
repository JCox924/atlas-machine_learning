#!/usr/bin/env python3
"""
Module one_hot contains function:
    one_hot(labels, classes=None)
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Args:
        labels: A vector of labels.
        classes: The total number of classes (optional).

    Returns:
        A one-hot encoded matrix.
    """
    one_hot_matrix = K.utils.to_categorical(labels,
                                            num_classes=classes)
    return one_hot_matrix
