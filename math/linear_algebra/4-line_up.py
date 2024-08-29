#!/usr/bin/env python3

def add_arrays(arr1, arr2):
    """

    :param arr1: array 1
    :param arr2: array 2
    :return: array of lists sum

    """

    sum_arr = []

    if len(arr1) != len(arr2):
        return None
    else:
        return [a + b for a, b in zip(arr1, arr2)]
