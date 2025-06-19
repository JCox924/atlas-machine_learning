#!/usr/bin/env python3
"""
2-from_file.py module
Defines a function from_file that loads a CSV file
    into a pandas DataFrame using a given delimiter.
"""

import pandas as pd


def from_file(filename, delimiter):
    """Loads data from a file into a pandas DataFrame.

    Parameters:
    filename : str
        The path to the file to load.
    delimiter : str
        The column separator used in the file.

    Returns:
        DataFrame loaded from the file.
    """
    return pd.read_csv(filename, sep=delimiter)
