#!/usr/bin/env python3
"""
Module convolve contains the function:
    convolve(images, kernels, padding='same', stride=(1, 1))
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Args:
        images (numpy.ndarray): with shape (m, h, w, c)
            containing multiple images.
            - m is the number of images.
            - h is the height in pixels of the images.
            - w is the width in pixels of the images.
            - c is the number of channels in the image.
        kernels (numpy.ndarray): with shape (kh, kw, c, nc)
            containing the kernels for the convolution.
            - kh is the height of a kernel.
            - kw is the width of a kernel.
            - nc is the number of kernels.
        padding (str or tuple): either a tuple of (ph, pw),
            'same', or 'valid'.
            - 'same': performs a same convolution.
            - 'valid': performs a valid convolution.
            - tuple: (ph, pw) is the padding for the
                height and width of the image.
        stride (tuple): (sh, sw) where:
            - sh is the stride for the height of the image.
            - sw is the stride for the width of the image.

    Returns:
        numpy.ndarray: containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    assert kc == c, ("The kernel must have the same"
                     " number of channels as the images")

    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded_imgs = np.pad(images,
                         ((0, 0),
                          (ph, ph),
                          (pw, pw),
                          (0, 0)),
                         mode='constant')

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    out = np.zeros((m, out_h, out_w, nc))

    for k in range(nc):
        for i in range(out_h):
            for j in range(out_w):
                vert_st = i * sh
                vert_e = vert_st + kh
                horiz_st = j * sw
                horiz_e = horiz_st + kw

                out[:, i, j, k] = np.sum(
                    padded_imgs[:, vert_st:vert_e, horiz_st:horiz_e, :]
                    * kernels[:, :, :, k],
                    axis=(1, 2, 3))

    return out
