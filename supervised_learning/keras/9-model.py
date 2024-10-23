#!/usr/bin/env python3
"""
Module model_io contains functions:
    save_model(network, filename)
    load_model(filename)
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model to a file.

    Args:
        network: The Keras model to save.
        filename: The path of the file that the model should be saved to.

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model from a file.

    Args:
        filename: The path of the file that the model should be loaded from.

    Returns:
        The loaded Keras model.
    """
    return K.models.load_model(filename)
