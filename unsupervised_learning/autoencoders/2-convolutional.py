#!/usr/bin/env python3
"""
Module: conv_autoencoder

This module provides a function to create a convolutional autoencoder.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Tuple of integers containing the dimensions of the model input
                            in the format (height, width, channels).
        filters (list): List of integers containing the number of filters for each
                        convolutional layer in the encoder.
        latent_dims (tuple): Tuple of integers containing the desired dimensions of the latent
                             space representation (height, width, channels).

    Returns:
        tuple: (encoder, decoder, auto)
            - encoder: the encoder model.
            - decoder: the decoder model.
            - auto: the full autoencoder model compiled with Adam optimizer and binary cross-entropy loss.
    """
    encoder_input = keras.Input(shape=input_dims, name="encoder_input")
    x = encoder_input
    for f in filters:
        x = keras.layers.Conv2D(filters=f,
                                kernel_size=(3, 3),
                                activation='relu',
                                padding='same')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    current_shape = keras.backend.int_shape(x)[1:3]
    target_shape = latent_dims[:2]
    if current_shape != target_shape:
        x = keras.layers.Resizing(target_shape[0], target_shape[1])(x)
    current_channels = keras.backend.int_shape(x)[-1]
    target_channels = latent_dims[-1]
    if current_channels != target_channels:
        x = keras.layers.Conv2D(filters=target_channels,
                                kernel_size=(1, 1),
                                activation='relu',
                                padding='same')(x)
    encoder_output = x
    encoder = keras.Model(encoder_input, encoder_output, name="encoder")

    decoder_input = keras.Input(shape=latent_dims, name="decoder_input")
    x = decoder_input
    rev_filters = filters[::-1]
    for i in range(len(rev_filters) - 1):
        x = keras.layers.Conv2D(filters=rev_filters[i],
                                kernel_size=(3, 3),
                                activation='relu',
                                padding='same')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
    x = keras.layers.Conv2D(filters=rev_filters[-1],
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='valid')(x)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)
    decoder_output = keras.layers.Conv2D(filters=input_dims[-1],
                                         kernel_size=(3, 3),
                                         activation='sigmoid',
                                         padding='same')(x)
    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    auto_input = encoder_input
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(auto_input, decoded, name="autoencoder")

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
