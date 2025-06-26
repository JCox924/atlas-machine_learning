#!/usr/bin/env python3
"""
13-analyze.py module
Defines a function `analyze` that computes descriptive statistics
    for all columns except 'Timestamp', without external imports.
"""

def analyze(df):
    """
    Computes descriptive statistics for
        all columns except the 'Timestamp' column.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing a 'Timestamp' column and other numeric columns.

    Returns:
    pandas.DataFrame
        DataFrame of descriptive statistics
            (count, mean, std, min, 25%, 50%, 75%, max)
        for all columns except 'Timestamp'.
    """
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    return df.describe()
