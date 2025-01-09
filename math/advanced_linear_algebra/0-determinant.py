#!/usr/bin/env python3
"""
Module 0-determinant contains function:
    determinant(matrix)
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix

    Arguments:
        matrix: must be list of lists
    Returns:
         Determinant of a matrix
    """

    if (not isinstance(matrix, list)
            or any(not isinstance(row, list) for row in matrix)):
        raise TypeError('matrix must be a list of lists')

    if matrix == [[]]:
        return 1

    n = len(matrix)

    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]

    if n == 2:
        # 2x2 -> AD - BC
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(n):
        submatrix = []
        for row_i in range(1, n):
            sub_row = matrix[row_i][:col] + matrix[row_i][col + 1:]
            submatrix.append(sub_row)

        sign = (-1) ** col

        det += sign * matrix[0][col] * determinant(submatrix)

    return det
