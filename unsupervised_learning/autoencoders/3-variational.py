#!/usr/bin/env python3
"""
Module: variational_autoencoder_module

This module provides a function to create a variational autoencoder (VAE).
The VAE consists of an encoder and a decoder. The encoder compresses the input
to a latent space representation using a series of hidden layers and outputs three tensors:
the sampled latent representation (using the reparameterization trick), the mean, and the log variance.
The decoder reconstructs the input from the latent representation using the reversed architecture of the encoder.
All hidden layers use ReLU activation except:
  - The mean and log variance layers in the encoder (which use no activation), and
  - The last layer in the decoder (which uses sigmoid activation).

The full autoencoder model is compiled with the Adam optimizer and binary cross-entropy loss.
"""

import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder (VAE).

    Args:
        input_dims (int): The dimensions of the model input.
        hidden_layers (list): A list containing the number of nodes for each hidden layer in the encoder.
        latent_dims (int): The dimensions of the latent space representation.

    Returns:
        tuple: (encoder, decoder, auto)
            - encoder is the encoder model that outputs the latent representation, mean, and log variance.
            - decoder is the decoder model.
            - auto is the full autoencoder model compiled with Adam optimizer and binary cross-entropy loss.
    """
    # --- Encoder ---
    encoder_input = keras.Input(shape=(input_dims,), name="encoder_input")
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(units=nodes, activation='relu')(x)

    # Mean and log variance layers (with no activation)
    z_mean = keras.layers.Dense(units=latent_dims, activation=None, name="z_mean")(x)
    z_log_var = keras.layers.Dense(units=latent_dims, activation=None, name="z_log_var")(x)

    # Reparameterization trick: sample latent vector from the learned distribution
    def sampling(args):
        mean, log_var = args
        epsilon = K.random_normal(shape=K.shape(mean))
        return mean + K.exp(0.5 * log_var) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,), name="z")([z_mean, z_log_var])

    # Encoder outputs: the latent vector, mean, and log variance
    encoder = keras.Model(encoder_input, [z, z_mean, z_log_var], name="encoder")

    # --- Decoder ---
    decoder_input = keras.Input(shape=(latent_dims,), name="decoder_input")
    x = decoder_input
    for nodes in hidden_layers[::-1]:
        x = keras.layers.Dense(units=nodes, activation='relu')(x)
    # Final layer with sigmoid activation to reconstruct the input
    decoder_output = keras.layers.Dense(units=input_dims, activation='sigmoid', name="decoder_output")(x)
    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    # --- Full Autoencoder ---
    auto_input = encoder_input
    z_sample, z_mean_out, z_log_var_out = encoder(auto_input)
    decoded = decoder(z_sample)
    auto = keras.Model(auto_input, decoded, name="autoencoder")

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
