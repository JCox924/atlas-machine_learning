#!/usr/bin/env python3
"""Module to perform hue adjustment on images."""
import tensorflow as tf


def change_hue(image, delta):
    """Adjusts the hue of a 3D image tensor by a given delta.

    Args:
        image (tf.Tensor): 3D tensor representing the image (H, W, C) with values in [0,1] or [0,255].
        delta (float): The amount to add to the hue channel. Should be in [-0.5, 0.5].

    Returns:
        tf.Tensor: The hue-adjusted image tensor.
    """
    return tf.image.adjust_hue(image, delta)
