#!/usr/bin/env python3
"""
    Calculates shape of a matrix

    Args:
        matrix (list): matrix or multi-dimensional array to find shape of
    Returns:
        list: list of ints reflecting the depth of that index
"""


def matrix_shape(matrix):
    shape = []

    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
