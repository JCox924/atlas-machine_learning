#!/usr/bin/env python3
"""
Module 5-dense_block contains function:
    dense_block(X, nb_filters, growth_rate, layers)
"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in 'Densely Connected Convolutional Networks'.

    Args:
        X: tensor, output from the previous layer
        nb_filters: int, number of filters in X
        growth_rate: int, growth rate for the dense block
        layers: int, number of layers in the dense block

    Returns:
        The concatenated output of each layer within the dense block
        The updated number of filters within the concatenated outputs
    """
    initializer = K.initializers.HeNormal(seed=0)

    for i in range(layers):
        bn1 = K.layers.BatchNormalization(axis=-1)(X)
        relu1 = K.layers.ReLU()(bn1)
        bottleneck = K.layers.Conv2D(4 * growth_rate,
                                     kernel_size=1,
                                     padding='same',
                                     kernel_initializer=initializer)(relu1)

        bn2 = K.layers.BatchNormalization(axis=-1)(bottleneck)
        relu2 = K.layers.ReLU()(bn2)
        conv = K.layers.Conv2D(growth_rate,
                               kernel_size=3,
                               padding='same',
                               kernel_initializer=initializer)(relu2)

        X = K.layers.Concatenate(axis=-1)([X, conv])

        nb_filters += growth_rate

    return X, nb_filters
