#!/usr/bin/env python3
"""
Module 0-create_confusion contains functions:
    create_confusion_matrix(labels, logits)
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Args:
        labels: one-hot matrix containing the correct labels
        logits: ont-hot matrix containing the network's predictions'
    Returns:
        confusion matrix
    """

    classes = labels.shape[1]

    confusion_matrix = np.zeros((classes, classes))

    actual = np.argmax(labels, axis=1)
    predictions = np.argmax(logits, axis=1)

    for i in range(len(actual)):
        confusion_matrix[actual[i], predictions[i]] += 1

    return confusion_matrix
