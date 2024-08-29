#!/usr/bin/env python3
"""matrix multiplication"""


def mat_mul(mat1, mat2):
    """
    :param mat1: matrix1
    :param mat2: matrix2
    :return: multiplied matrix
    """

    if len(mat1[0]) != len(mat2):
        return None
    else:
        mat2_T = list(zip(*mat2))

        return [[sum(a * b for a, b in zip(row, col)) for col in mat2_T] for row in mat1]
