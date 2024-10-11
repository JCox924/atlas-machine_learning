#!/usr/bin/env python3
"""
Module contains functions:
    f1_score(confusion)
"""


def f1_score(confusion):
    """
    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
            represent the correct labels and
                column indices represent the predicted labels
                - classes: number of classes
    Returns:
        numpy.ndarray of shape (classes,) containing the f1 score of each class
    """
