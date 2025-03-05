#!/usr/bin/env python3
"""
Module provides a function to create an autoencoder model.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder model with the specified architecture.

    Args:
        input_dims (int): The dimensions of the model input.
        hidden_layers (list): A list of integers, each representing the number
                              of nodes in each hidden layer of the encoder.
        latent_dims (int): The dimensions of the latent space representation.

    Returns:
        tuple: A tuple (encoder, decoder, auto) where:
            - encoder is the encoder model.
            - decoder is the decoder model.
            - auto is the full autoencoder model compiled with the Adam optimizer
              and binary cross-entropy loss.
    """
    # Build the encoder model
    inputs = keras.Input(shape=(input_dims,), name="encoder_input")
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu', name="latent")(x)
    encoder = keras.Model(inputs, latent, name="encoder")

    # Build the decoder model using the reversed hidden layers
    latent_inputs = keras.Input(shape=(latent_dims,), name="decoder_input")
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid', name="decoder_output")(x)
    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # Combine encoder and decoder into the autoencoder model
    auto_inputs = inputs
    encoded = encoder(auto_inputs)
    decoded = decoder(encoded)
    auto = keras.Model(auto_inputs, decoded, name="autoencoder")

    # Compile the autoencoder with Adam optimizer and binary cross-entropy loss
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
