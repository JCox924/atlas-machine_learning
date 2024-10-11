#!/usr/bin/env python3
"""
Module 1-sensitivity contains functions:
    sensitivity(confusion)
"""
import numpy as np


def sensitivity(confusion):
    """
    Args:
        confusion: confusion matrix of shape (classes, classes)
    Returns:
        np.array of shape (classes, ) containing the sensitivity for each class
    """

    sens_matrix = np.zeros(confusion.shape[0])

    for i in range(confusion.shape[0]):
        true_positives = confusion[i, i]
        false_negatives = np.sum(confusion[i, :]) - true_positives
        sens_matrix[i] = true_positives / (true_positives + false_negatives)

    return sens_matrix
