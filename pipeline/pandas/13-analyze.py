#!/usr/bin/env python3
"""
12-hierarchy.py module
Defines a function `hierarchy` that:
- Indexes two DataFrames on 'Timestamp'.
- Selects rows from both DataFrames between
    timestamps 1417411980 and 1417417980 inclusive.
- Concatenates the selected bitstamp (df2) above
    coinbase (df1) data with keys.
- Rearranges the MultiIndex so that 'Timestamp' is the
    first level and the service key is the second.
- Returns the concatenated DataFrame sorted
    in chronological order by timestamp.
"""

import pandas as pd


def hierarchy(df1, df2):
    """
    Combines bitstamp and coinbase data in a hierarchical index by timestamp.

    Parameters:
    df1 : pandas.DataFrame
        Coinbase data containing a 'Timestamp' column.
    df2 : pandas.DataFrame
        Bitstamp data containing a 'Timestamp' column.

    Returns:
    pandas.DataFrame
        MultiIndexed DataFrame with levels ('Timestamp', service key),
            containing rows from both sources
            between timestamps 1417411980 and 1417417980, sorted chronologically.
    """
    index = __import__('10-index').index

    df1_i = index(df1).loc[1417411980:1417417980]
    df2_i = index(df2).loc[1417411980:1417417980]

    combined = pd.concat([df2_i, df1_i], keys=['bitstamp', 'coinbase'])

    hier = combined.swaplevel(0, 1)

    return hier.sort_index(level=0)
