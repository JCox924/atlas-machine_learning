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
    f1_scores = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        sens = sensitivity(confusion)[i]
        prec = precision(confusion)[i]

        if prec + sens == 0:
            f1_scores[i] = 0
        else:
            f1_scores[i] = 2 * (prec * sens)/(prec + sens)

    return f1_scores
