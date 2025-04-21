#!/usr/bin/env python3
"""
4-create_masks.py

Creates the padding and look-ahead masks needed for
training/validation of the Transformer.
"""

import tensorflow as tf


def create_masks(inputs, target):
    """
    Generates all masks required for the Transformer’s attention blocks.

    Args:
        inputs (tf.Tensor): shape (batch_size, seq_len_in), input token IDs.
        target (tf.Tensor): shape (batch_size, seq_len_out), target token IDs.

    Returns:
        encoder_mask (tf.Tensor): shape (batch_size, 1, 1, seq_len_in);
            padding mask for the encoder.
        combined_mask (tf.Tensor): shape
         (batch_size, 1, seq_len_out, seq_len_out);
            padding + look-ahead mask for the
            decoder’s first attention block.
        decoder_mask (tf.Tensor): shape (batch_size, 1, 1, seq_len_in);
            padding mask for the decoder’s second attention block.
    """
    enc_padding_mask = tf.cast(tf.equal(inputs, 0), tf.float32)
    enc_padding_mask = enc_padding_mask[:, tf.newaxis, tf.newaxis, :]

    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out), dtype=tf.float32),
        num_lower=-1,
        num_upper=0
    )

    dec_target_padding_mask = tf.cast(tf.equal(target, 0), tf.float32)
    dec_target_padding_mask = (
                                  dec_target_padding_mask)[:,
                              tf.newaxis,
                              tf.newaxis, :]

    combined_mask = tf.maximum(
        dec_target_padding_mask,
        look_ahead_mask[tf.newaxis, tf.newaxis, :, :]
    )

    dec_padding_mask = enc_padding_mask

    return enc_padding_mask, combined_mask, dec_padding_mask
