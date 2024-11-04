#!/usr/bin/env python3
"""
Module 0-inception_block contains funtions inception_block(A_prev, filters):
"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in 'Going Deeper with Convolutions' (2014).

    Args:
        A_prev: tensor, output from the previous layer
        filters: tuple or list, contains F1, F3R, F3, F5R, F5, FPP
            F1: filters in the 1x1 convolution
            F3R: filters in the 1x1 convolution before the 3x3 convolution
            F3: filters in the 3x3 convolution
            F5R: filters in the 1x1 convolution before the 5x5 convolution
            F5: filters in the 5x5 convolution
            FPP: filters in the 1x1 convolution after max pooling

    Returns:
        concatenated: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    conv1x1 = K.layers.Conv2D(filters=F1, kernel_size=1, padding='same', activation='relu')(A_prev)

    conv3x3_reduce = K.layers.Conv2D(filters=F3R, kernel_size=1, padding='same', activation='relu')(A_prev)
    conv3x3 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same', activation='relu')(conv3x3_reduce)

    conv5x5_reduce = K.layers.Conv2D(filters=F5R, kernel_size=1, padding='same', activation='relu')(A_prev)
    conv5x5 = K.layers.Conv2D(filters=F5, kernel_size=5, padding='same', activation='relu')(conv5x5_reduce)

    max_pool = K.layers.MaxPooling2D(pool_size=3, strides=1, padding='same')(A_prev)
    max_pool_conv = K.layers.Conv2D(filters=FPP, kernel_size=1, padding='same', activation='relu')(max_pool)

    concat = K.layers.Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5, max_pool_conv])

    return concat
