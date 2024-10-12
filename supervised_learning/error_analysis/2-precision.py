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
        numpy.ndarray of shape (classes,)
            containing the precision of each class
    """
    precision_matrix = np.zeros(confusion.shape[0])

    for i in range(confusion.shape[0]):
        true_pos = confusion[i, i]
        false_pos = np.sum(confusion[:, i]) - true_pos
        precision_matrix[i] = true_pos / (true_pos + false_pos)

    return precision_matrix
