#!/usr/bin/env python3
"""Module contains function moving_average."""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a dataset using bias correction.

    Args:
        data: The list of data points.
        beta: The weight used for the moving average.

    Returns:
        list: A list containing the moving averages of the data.
    """
    vt = 0
    averages = []

    for i, value in enumerate(data):
        vt = beta * vt + (1 - beta) * value
        bias_correction = 1 - beta ** (i + 1)
        corrected_vt = vt / bias_correction
        averages.append(corrected_vt)

    return averages
