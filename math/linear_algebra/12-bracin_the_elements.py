#!/usr/bin/env python3

"""embracing the elements """


def np_elementwise(mat1, mat2):
    """

    :param mat1: matrix1
    :param mat2: matrix2
    :return: tuple of the sums, diff, products and divisons

    """
    result = {
        'add': mat1 + mat2,
        'sub': mat1 - mat2,
        'mul': mat1 * mat2,
        'div': np.divide(mat1, mat2)
    }

    return result