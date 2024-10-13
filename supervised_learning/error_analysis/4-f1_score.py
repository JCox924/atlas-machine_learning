#!/usr/bin/env python3
"""
Module contains functions:
    f1_score(confusion)
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
            represent the correct labels and
                column indices represent the predicted labels
                - classes: number of classes
    Returns:
        numpy.ndarray of shape (classes) containing the f1 score of each class
    """
    con = confusion
    f1_scores = np.zeros(con.shape[0])
    for i in range(con.shape[0]):
        f1_scores[i] = 2 * sensitivity(con) * precision(con) / (sensitivity(con) + precision(con))
    return f1_scores
