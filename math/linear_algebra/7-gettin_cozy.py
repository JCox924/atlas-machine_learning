#!/usr/bin/env python3

"""concats matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """

    :param mat1: 2D array
    :param mat2: 2D array
    :param axis: axis to concat matrices
    :return: new combined matrix

    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None