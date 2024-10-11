#!/usr/bin/env python3
"""
Module 3-precision contains funtions:
    specificity(confusion)
"""


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

    specificity_matrix = confusion.astype(float) / confusion.sum(axis=1, keepdims=True)

    return specificity_matrix
