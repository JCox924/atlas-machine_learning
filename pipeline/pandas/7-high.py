#!/usr/bin/env python3
"""
Defines a function `high` that sorts a pandas DataFrame
    by its 'High' column in descending order.
"""


def high(df):
    """Sorts the DataFrame by 'High' in descending order.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing a 'High' column.

    Returns:
    pandas.DataFrame
        DataFrame sorted by 'High' descending.
    """
    return df.sort_values(by='High', ascending=False)
