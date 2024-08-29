#!/usr/bin/env python3

def matrix_transpose(matrix):
    """
    :param matrix: 2D array to find the transpose of
    :return: matrix T
    """
    transpose = [[row[i] for row in matrix] for i in range(len(matrix[0]))]

    return transpose
