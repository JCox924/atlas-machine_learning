#!/usr/bin/env python3
"""
Module 0-sequential contains function:
    build_model(nx, layers, activations, lambtha, keep_prob)
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a model with layers and activation functions using the Keras functional API.

    Args:
        nx: The number of input features to the network.
        layers: A list containing the number of nodes in each layer of the network.
        activations: A list containing the activation functions for each layer.
        lambtha: The L2 regularization parameter.
        keep_prob: The keep probability for the dropout layer.

    Returns: Keras model
    """
    inputs = K.Input(shape=(nx,))
    reg = K.regularizers.l2(lambtha)

    x = K.layers.Dense(layers[0],
                       activation=activations[0],
                       kernel_regularizer=reg)(inputs)

    for i in range(1, len(layers)):
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(layers[i],
                           activation=activations[i],
                           kernel_regularizer=reg)(x)

    model = K.Model(inputs=inputs, outputs=x)

    return model
