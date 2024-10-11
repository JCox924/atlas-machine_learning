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

    precision = np.diag(confusion)

    precision_matrix = precision.reshape((confusion.shape[0], confusion.shape[1]))

    return precision_matrix
