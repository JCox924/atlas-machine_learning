#!/usr/bin/env python3
"""
Module: sparse_autoencoder_module

This module provides a function to create a sparse autoencoder model.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder model.

    Args:
        input_dims (int): The dimensions of the model input.
        hidden_layers (list): A list of integers representing the number of nodes
                              for each hidden layer in the encoder.
        latent_dims (int): The dimensions of the latent space representation.
        lambtha (float): The regularization parameter for L1 regularization applied
                         on the encoded output.

    Returns:
        tuple: A tuple (encoder, decoder, auto) where:
            - encoder is the encoder model.
            - decoder is the decoder model.
            - auto is the sparse autoencoder model compiled with Adam optimizer
              and binary cross-entropy loss.
    """
    inputs = keras.Input(shape=(input_dims,), name="encoder_input")
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha),
        name="latent"
    )(x)
    encoder = keras.Model(inputs, latent, name="encoder")

    latent_inputs = keras.Input(shape=(latent_dims,), name="decoder_input")
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid', name="decoder_output")(x)
    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # Full autoencoder model
    auto_inputs = inputs
    encoded = encoder(auto_inputs)
    decoded = decoder(encoded)
    auto = keras.Model(auto_inputs, decoded, name="autoencoder")

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
