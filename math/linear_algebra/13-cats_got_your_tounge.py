#!/usr/bin/env python3

"""concatonation"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """

    :param mat1: matrix1
    :param mat2: matrix2
    :param axis: axis to concat
    :return: new matrix

    """
    return np.concatenate((mat1, mat2), axis=axis)
