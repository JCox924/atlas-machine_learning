#!/usr/bin/env python3

"""adds two matricies"""


def matrix_shape(matrix):
    """
        Calculates shape of a matrix

        Args:
            matrix (list): matrix or multi-dimensional array to find shape of
        Returns:
            list: list of ints reflecting the depth of that index
    """

    shape = []

    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape


def add_matrices2D(mat1, mat2):
    """

    :param mat1: matrix
    :param mat2: matrix
    :return: sum of matrices

    """

    if matrix_shape(mat1) != matrix_shape(mat2) or not mat1 and mat2:
        return None
    else:
        return [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]
