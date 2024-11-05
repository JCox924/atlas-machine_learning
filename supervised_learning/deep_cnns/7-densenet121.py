#!/usr/bin/env python3
"""
Module 7-densenet121 contains function:
    densenet121(growth_rate=32, compression=1.0)
"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in
        'Densely Connected Convolutional Networks'.

    Args:
        growth_rate: int, growth rate for the dense blocks
        compression: float, compression factor
            for the transition layers (0 < compression <= 1)

    Returns:
        model: the keras Model
    """
    initializer = K.initializers.HeNormal(seed=0)

    input_layer = K.Input(shape=(224, 224, 3))

    # Initial convolution and max pooling
    X = K.layers.BatchNormalization(axis=-1)(input_layer)
    X = K.layers.Conv2D(64,
                        kernel_size=7,
                        strides=2,
                        padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.ReLU()(X)
    X = K.layers.MaxPooling2D(pool_size=3,
                              strides=2,
                              padding='same')(X)

    # Dense Block 1
    X, nb_filters = dense_block(X,
                                nb_filters=64,
                                growth_rate=growth_rate,
                                layers=6)
    # Transition Layer 1
    X, nb_filters = transition_layer(X,
                                     nb_filters=nb_filters,
                                     compression=compression)

    # Dense Block 2
    X, nb_filters = dense_block(X,
                                nb_filters=nb_filters,
                                growth_rate=growth_rate,
                                layers=12)
    # Transition Layer 2
    X, nb_filters = transition_layer(X,
                                     nb_filters=nb_filters,
                                     compression=compression)

    # Dense Block 3
    X, nb_filters = dense_block(X,
                                nb_filters=nb_filters,
                                growth_rate=growth_rate,
                                layers=24)
    # Transition Layer 3
    X, nb_filters = transition_layer(X,
                                     nb_filters=nb_filters,
                                     compression=compression)

    # Dense Block 4
    X, nb_filters = dense_block(X,
                                nb_filters=nb_filters,
                                growth_rate=growth_rate,
                                layers=16)

    # Classification Layer
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.ReLU()(X)
    X = K.layers.GlobalAveragePooling2D()(X)
    output = K.layers.Dense(1000,
                            activation='softmax',
                            kernel_initializer=initializer)(X)

    model = K.Model(inputs=input_layer, outputs=output)

    return model
