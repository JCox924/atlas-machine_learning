#!/usr/bin/env python3
"""
Module 2-cofactor contains:
    functions:
        cofactor(matrix)
"""


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a square matrix.

    Parameters:
    - matrix: a list of lists whose cofactor matrix should be calculated

    Raises:
    - TypeError: if matrix is not a list of lists
    - ValueError: if matrix is empty or not square

    Returns:
    - The cofactor matrix of matrix
    """
    if (not isinstance(matrix, list) or len(matrix) == 0
            or any(not isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)  # number of rows
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    def determinant(m):
        """Compute the determinant of a square matrix m (list of lists)."""
        size = len(m)

        if size == 1:
            return m[0][0]
        if size == 2:
            return m[0][0]*m[1][1] - m[0][1]*m[1][0]

        det = 0
        for j in range(size):
            sub_matrix = [row[:j] + row[j+1:] for row in m[1:]]
            det += ((-1) ** j) * m[0][j] * determinant(sub_matrix)
        return det

    def get_submatrix(m, i, j):
        """
        Return the submatrix of m obtained by removing the i-th row
        and the j-th column.
        """
        return [row[:j] + row[j+1:] for idx, row in enumerate(m) if idx != i]

    cof_matrix = []
    for i in range(n):
        cof_row = []
        for j in range(n):
            minor_det = determinant(get_submatrix(matrix, i, j))
            cof_value = ((-1)**(i+j)) * minor_det
            cof_row.append(cof_value)
        cof_matrix.append(cof_row)

    return cof_matrix
