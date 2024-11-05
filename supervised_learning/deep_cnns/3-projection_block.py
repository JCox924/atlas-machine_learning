#!/usr/bin/env python3
"""
Module 3-projection_block contains functions:
    projection_block(A_prev, filters, s=2)
"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in
        'Deep Residual Learning for Image Recognition' (2015).

    Args:
        A_prev: tensor, output from the previous layer
        filters: tuple or list, contains F11, F3, F12
            F11: filters in the first 1x1 convolution
            F3: filters in the 3x3 convolution
            F12: filters in the second 1x1 convolution and the shortcut path
        s: int, stride for the first convolution
            in both main path and shortcut path

    Returns:
        activated output of the projection block
    """
    F11, F3, F12 = filters
    initializer = K.initializers.HeNormal(seed=0)

    x = K.layers.Conv2D(filters=F11,
                        kernel_size=1,
                        strides=s,
                        padding='same',
                        kernel_initializer=initializer)(A_prev)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.ReLU()(x)

    x = K.layers.Conv2D(filters=F3,
                        kernel_size=3,
                        padding='same',
                        kernel_initializer=initializer)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.ReLU()(x)

    x = K.layers.Conv2D(filters=F12,
                        kernel_size=1,
                        padding='same',
                        kernel_initializer=initializer)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)

    shortcut = K.layers.Conv2D(filters=F12,
                               kernel_size=1,
                               strides=s,
                               padding='same',
                               kernel_initializer=initializer)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=-1)(shortcut)

    x = K.layers.Add()([x, shortcut])
    output = K.layers.ReLU()(x)

    return output
