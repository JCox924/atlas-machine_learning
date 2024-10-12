#!/usr/bin/env python3
"""
Module 2-precision contains functions:
    precision(confusion)
"""
import numpy as np


def precision(confusion):
    """
    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
            represent the correct labels and
                column indices represent the predicted labels
                - classes: number of classes
    Returns:
        numpy.ndarray of shape (classes,) containing the precision of each class
    """
    precision_matrix = np.zeros_like(confusion)

    for i in range(confusion.shape[0]):
        true_pos = np.sum(confusion[i, i])
        false_pos = np.sum(confusion[:, i])
        precision_matrix[i, i] = true_pos / (true_pos + false_pos)

    return precision_matrix
