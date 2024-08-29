#!/usr/bin/env python3

"""adds two matricies"""


def add_matrices2D(mat1, mat2):
    """

    :param mat1: matrix
    :param mat2: matrix
    :return: sum of matrices

    """

    if len(mat1) != len(mat2):
        return None
    else:
        return [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]
