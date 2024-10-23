#!/usr/bin/env python3
"""
Module convolve contains the function:
    convolve_grayscale_padding(images, kernel, padding)
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
        images (numpy.ndarray): with shape (m, h, w)
            containing multiple grayscale images.
            - m is the number of images.
            - h is the height in pixels of the images.
            - w is the width in pixels of the images.
        kernel (numpy.ndarray): with shape (kh, kw) containing
            the kernel for the convolution.
            - kh is the height of the kernel.
            - kw is the width of the kernel.
        padding (tuple): (ph, pw) where:
            - ph is the padding for the height of the image.
            - pw is the padding for the width of the image.

    Returns:
        numpy.ndarray: containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded_images = np.pad(images,
                           ((0, 0),
                            (ph, ph),
                            (pw, pw)),
                           mode='constant')

    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    convolved_images = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            convolved_images[:, i, j] = np.sum(padded_images[:, i:i+kh, j:j+kw]
                                               * kernel, axis=(1, 2))

    return convolved_images
