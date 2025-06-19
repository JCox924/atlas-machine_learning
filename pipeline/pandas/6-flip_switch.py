#!/usr/bin/env python3
"""
Defines a function flip_switch that sorts a DataFrame
    in reverse chronological order by the 'Timestamp' column,
    transposes the result, and returns it.
"""


def flip_switch(df):
    """
    Sorts the DataFrame in reverse chronological order by
        'Timestamp', transposes it, and returns it.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing a 'Timestamp' column.

    Returns:
    pandas.DataFrame
        The transposed DataFrame after sorting by 'Timestamp' descending.
    """
    df_sorted = df.sort_values('Timestamp', ascending=False)
    return df_sorted.T
