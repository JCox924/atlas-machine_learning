#!/usr/bin/env python3
"""
Module convolve_channels contains the function:
    convolve_channels(images, kernel, padding='same', stride=(1, 1))
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with multiple channels.

    Args:
        images (numpy.ndarray): with shape (m, h, w, c) containing multiple images.
            - m is the number of images.
            - h is the height in pixels of the images.
            - w is the width in pixels of the images.
            - c is the number of channels in the image.
        kernel (numpy.ndarray): with shape (kh, kw, c) containing the kernel for the convolution.
            - kh is the height of the kernel.
            - kw is the width of the kernel.
            - c is the number of channels, which must match the input images.
        padding (str or tuple): either a tuple of (ph, pw), 'same', or 'valid'.
            - 'same': performs a same convolution.
            - 'valid': performs a valid convolution.
            - tuple: (ph, pw) is the padding for the height and width of the image.
        stride (tuple): (sh, sw) where:
            - sh is the stride for the height of the image.
            - sw is the stride for the width of the image.

    Returns:
        numpy.ndarray: containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    assert kc == c, "The kernel must have the same number of channels as the images"

    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            output[:, i, j] = np.sum(
                padded_images[:, vert_start:vert_end, horiz_start:horiz_end, :] * kernel, axis=(1, 2, 3))

    return output
