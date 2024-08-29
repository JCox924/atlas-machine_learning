#!/usr/bin/env python3
"""find size of matrix"""
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
