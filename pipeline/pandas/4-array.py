#!/usr/bin/env python3
"""
Defines a function array that selects the last 10 rows of 'High' and 'Close'
    columns from a pandas DataFrame and returns them as a numpy ndarray.
"""

def array(df):
    """
    Selects the last 10 rows of 'High' and 'Close' columns and
        returns them as a numpy.ndarray.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing 'High' and 'Close' columns.

    Returns:
        Array of shape (10, 2) with the selected values.
    """
    return df[['High', 'Close']].tail(10).values
