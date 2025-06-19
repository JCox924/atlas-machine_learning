#!/usr/bin/env python3
"""
Defines a function prune that removes rows with NaN
    values in the 'Close' column of a pandas DataFrame.
"""


def prune(df):
    """Removes any entries where 'Close' has NaN values.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing a 'Close' column.

    Returns:
    pandas.DataFrame
        DataFrame with rows dropped where 'Close' is NaN.
    """
    return df.dropna(subset=['Close'])
