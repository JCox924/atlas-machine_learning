#!/usr/bin/env python3
"""
Module config_io contains functions:
    save_config(network, filename)
    load_config(filename)
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format.

    Args:
        network: The Keras model whose configuration should be saved.
        filename: The path of the file that the
            configuration should be saved to.

    Returns:
        None
    """
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)


def load_config(filename):
    """
    Loads a model with a specific configuration from a JSON file.

    Args:
        filename: The path of the file containing the model's
            configuration in JSON format.

    Returns:
        The loaded Keras model.
    """
    with open(filename, 'r') as f:
        config = f.read()

    model = K.models.model_from_json(config)
    return model
