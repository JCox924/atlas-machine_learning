#!/usr/bin/env python3
"""
Module 0-sequential contains function:
    build_model(nx, layers, activations, lambtha, keep_prop)
"""
import tensorflow.keras as K
def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a squential model with layers and activation functions.

    Args:
        nx: The number of input features to the network.
        layers: A list containing the number of nodes in each layer of the network.
        activations: A list containing the activation functions for each layer.
        lambtha: The L2 regularization parameter.
        keep_prob: The keep probability for the dropout layer.
    Returns: Keras model
    """

    model = K.Sequential()
    reg = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=reg,
                                     input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=reg))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
