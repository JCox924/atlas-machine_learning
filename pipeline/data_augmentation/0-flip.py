#!/usr/bin/env python3
"""Module to flip images horizontally."""
import tensorflow as tf


def flip_image(image):
    """Flips a 3D image tensor horizontally.

    Args:
        image (tf.Tensor): 3D tensor representing the image (H, W, C).

    Returns:
        tf.Tensor: The horizontally flipped image tensor.
    """
    return tf.image.flip_left_right(image)
