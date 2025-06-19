#!/usr/bin/env python3
"""
Defines a function `fill` that cleans and fills
    missing data in a pandas DataFrame.
"""


def fill(df):
    """
    Removes 'Weighted_Price', fills missing values
        according to specified rules.

    Parameters:
        DataFrame containing columns
            'Weighted_Price', 'Close', 'High', 'Low', 'Open',
            'Volume_(BTC)', and 'Volume_(Currency)'.

    Returns:
        Modified DataFrame with 'Weighted_Price'
            dropped, missing values filled:
        - 'Close': forward-filled from previous row
        - 'High', 'Low', 'Open': filled with
            corresponding 'Close' value
        - 'Volume_(BTC)', 'Volume_(Currency)': filled with 0
    """
    if 'Weighted_Price' in df.columns:
        df = df.drop(columns=['Weighted_Price'])

    df['Close'] = df['Close'].fillna(method='ffill')

    for col in ['Open', 'High', 'Low']:
        df[col] = df[col].fillna(df['Close'])

    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
