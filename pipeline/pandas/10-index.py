#!/usr/bin/env python3
"""
Defines a function `index` that sets the 'Timestamp'
    column as the index of a pandas DataFrame.
"""


def index(df):
    """
    Sets 'Timestamp' as the DataFrame index.

    Parameters:
        DataFrame containing a 'Timestamp' column.

    Returns:
        Modified DataFrame with 'Timestamp' as the index.
    """
    return df.set_index('Timestamp')
