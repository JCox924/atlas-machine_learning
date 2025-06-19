#!/usr/bin/env python3
"""
Defines a function `slice` that extracts selected columns
    and every 60th row from a pandas DataFrame.
"""


def slice(df):
    """Selects specific columns and every 60th row from the DataFrame.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing at least the columns
            'High', 'Low', 'Close', and 'Volume_(BTC)'.

    Returns:
    pandas.DataFrame
        Sliced DataFrame with only the specified columns and every 60th row.
    """
    cols = ['High', 'Low', 'Close', 'Volume_(BTC)']
    sub_df = df[cols]
    return sub_df.iloc[::60]
