#!/usr/bin/env python3
"""
Module 1-minor contains:
    functions:
        minor(matrix)
"""


def minor(matrix):
    """
    Calculates the minor of a square matrix.

    Arguments:
        matrix: must be a square list of lists
    Returns:
         array: minor of the matrix
    """

    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise TypeError('matrix must be a non-empty square matrix')

    def determinant(m):
        """Compute the determinant of a square matrix m (list of lists)."""
        size = len(m)

        if size == 1:
            return m[0][0]
        if size == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]

        det = 0
        for j in range(size):
            sub_matrix = [row[:j] + row[j + 1:] for row in m[1:]]
            det += ((-1) ** j) * m[0][j] * determinant(sub_matrix)
        return det

    minor_matrix = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix)):
            sub_matrix = [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)

    return minor_matrix
