#!/usr/bin/env python3
"""
Defines a function `rename` that renames the Timestamp column to Datetime,
    converts it to datetime objects, and returns
    only the Datetime and Close columns.
"""
import pandas as pd


def rename(df):
    """Renames 'Timestamp' column to 'Datetime', converts to datetime, and filters columns.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing a 'Timestamp' column and a 'Close' column.

    Returns:
        Modified DataFrame with only 'Datetime' (as datetime objects) and 'Close'.
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    return df[['Datetime', 'Close']]
