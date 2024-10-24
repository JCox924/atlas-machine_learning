#!/usr/bin/env python3
"""
Module test_model contains function:
    test_model(network, data, labels, verbose=True)
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network.

    Args:
        network: The network model to test.
        data: A numpy.ndarray of shape (m, nx) containing
            the input data to test the model with.
        labels: A one-hot numpy.ndarray of shape (m, classes)
            containing the correct labels of data.
        verbose: A boolean that determines if output should be
            printed during the testing process.

    Returns:
        tuple: The loss and accuracy of the model
            with the testing data, respectively.
    """
    loss, accuracy = network.evaluate(data, labels, verbose=verbose)

    return [loss, accuracy]
