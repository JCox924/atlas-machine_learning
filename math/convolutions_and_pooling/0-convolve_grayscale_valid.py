#!/usr/bin/env python3
"""
Module convolve contains the function:
    convolve_grayscale_valid(images, kernel)
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
        images (numpy.ndarray): with shape (m, h, w) containing multiple grayscale images.
            - m is the number of images.
            - h is the height in pixels of the images.
            - w is the width in pixels of the images.
        kernel (numpy.ndarray): with shape (kh, kw) containing the kernel for the convolution.
            - kh is the height of the kernel.
            - kw is the width of the kernel.

    Returns:
        numpy.ndarray: containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_h = h - kh + 1
    output_w = w - kw + 1

    convolved_images = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            convolved_images[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return convolved_images
