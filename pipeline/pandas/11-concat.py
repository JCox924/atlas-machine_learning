#!/usr/bin/env python3
"""
Defines a function `concat` that indexes two DataFrames on 'Timestamp',
    selects bitstamp rows up to a given timestamp,
    and concatenates with keys.
"""
import pandas as pd


def concat(df1, df2):
    """
    Concatenates bitstamp data up to timestamp 1417411920 on top
    of coinbase data, using 'Timestamp' as
    index and labeling keys.

    Parameters:
        df1 : DataFrame for coinbase data containing a 'Timestamp' column.
        df2 : DataFrame for bitstamp data containing a 'Timestamp' column.

    Returns:
        MultiIndexed DataFrame with keys 'bitstamp' and 'coinbase'.
    """
    index = __import__('10-index').index

    df1_i = index(df1)
    df2_i = index(df2)

    df2_sel = df2_i.loc[:1417411920]

    return pd.concat([df2_sel, df1_i], keys=['bitstamp', 'coinbase'])
