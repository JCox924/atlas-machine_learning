#!/usr/bin/env python3
"""Defines a function from_numpy that creates a pandas DataFrame
    from a numpy array
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pandas DataFrame from a numpy array

    :param array: Input array.
    :return: Converted array.
    """

    n_cols = array.shape[1]
    labels = [chr(ord('A') + i) for i in range(n_cols)]
    return pd.DataFrame(array, columns=labels)
