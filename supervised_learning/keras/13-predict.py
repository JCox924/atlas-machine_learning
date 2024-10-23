#!/usr/bin/env python3
"""
Module predict contains function:
    predict(network, data, verbose=False)
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.

    Args:
        network: The network model to make the prediction with.
        data: A numpy.ndarray of shape (m, nx) containing the input data to make the prediction with.
        verbose: A boolean that determines if output should be printed during the prediction process.

    Returns:
        numpy.ndarray: The prediction for the data.
    """
    predictions = network.predict(data, verbose=verbose)
    return predictions
