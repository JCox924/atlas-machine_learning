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

    if not isinstance(matrix, list) or not all(isinstance(row,list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise TypeError('matrix must be a non-empty square matrix')

    determinant = __import__('0-determinant').determinant

    minor_matrix = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix)):
            sub_matrix = [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)

    return minor_matrix
