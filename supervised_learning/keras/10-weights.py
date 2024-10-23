#!/usr/bin/env python3
"""
Module weights_io contains functions:
    save_weights(network, filename, save_format='keras')
    load_weights(network, filename)
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights to a file.

    Args:
        network: The Keras model whose weights should be saved.
        filename: The path of the file that the weights should be saved to.
        save_format: The format in which the weights should be saved ('keras' or 'h5').

    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads a model's weights from a file.

    Args:
        network: The Keras model to which the weights should be loaded.
        filename: The path of the file that the weights should be loaded from.

    Returns:
        None
    """
    network.load_weights(filename)
