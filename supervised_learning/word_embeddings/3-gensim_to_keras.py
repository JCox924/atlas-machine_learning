#!/usr/bin/env python3
"""
Module that converts a gensim Word2Vec model to a Keras Embedding layer.

This module provides the function gensim_to_keras which extracts the weight matrix from a
trained gensim Word2Vec model and creates a Keras Embedding layer initialized with these weights.
The embedding layer is trainable.
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim Word2Vec model to a Keras Embedding layer.

    The function extracts the word vectors from the gensim model (using model.wv.vectors)
    and creates a Keras Embedding layer using these weights. The embedding layer is trainable,
    so the weights can be further updated during training.

    Args:
        model: A trained gensim Word2Vec model.

    Returns:
        tensorflow.keras.layers.Embedding: A trainable Keras Embedding layer initialized with
        the gensim model's weights.
    """
    _ = model.wv.index_to_key
    weights = model.wv.vectors
    vocab_size, embedding_dim = weights.shape

    keras_embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                output_dim=embedding_dim,
                                weights=[weights],
                                trainable=True)
    return keras_embedding
