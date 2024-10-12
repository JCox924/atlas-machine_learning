#!/usr/bin/env python3
"""
Module 3-precision contains functions:
    specificity(confusion)
"""
import numpy as np


def specificity(confusion):
    """
    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
            represent the correct labels and
                column indices represent the predicted labels
                - classes: number of classes
    Returns:
        numpy.ndarray of shape (classes,) containing the specificity of each class
    """
    specificity_matrix = np.zeros_like(confusion)
    total = np.sum(confusion)

    for i in range(confusion.shape[0]):

        true_pos = confusion[i][i]
        false_pos = np.sum(confusion[:, i]) - true_pos
        false_neg = np.sum(confusion[i, :]) - true_pos

        true_neg = total - (true_pos + false_pos + false_neg)

        specificity_matrix[i] = true_neg / (true_neg + false_pos)

    return specificity_matrix
