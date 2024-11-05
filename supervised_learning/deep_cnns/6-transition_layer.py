#!/usr/bin/env python3
"""
Module 6-transition_layer contains function:
    transition_layer(X, nb_filters, compression)
"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in
        'Densely Connected Convolutional Networks'.

    Args:
        X: tensor, output from the previous layer
        nb_filters: int, number of filters in X
        compression: float, compression factor
            for the transition layer (0 < compression <= 1)

    Returns:
        The output of the transition layer
        The updated number of filters within the output
    """
    initializer = K.initializers.HeNormal(seed=0)

    nb_filters = int(nb_filters * compression)

    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.ReLU()(X)
    X = K.layers.Conv2D(nb_filters,
                        kernel_size=1,
                        padding='same',
                        kernel_initializer=initializer)(X)

    X = K.layers.AveragePooling2D(pool_size=2,
                                  strides=2,
                                  padding='same')(X)

    return X, nb_filters
