#!/usr/bin/env python3
"""
Module for performing back propagation over a convolutional layer
"""
import numpy as np


def conv_backward(dZ, A_prev, W, padding="same", stride=(1, 1), mode='max'):
    """
    Performs back propagation over a convolutional layer
    Args:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new)
            containing partial derivatives with respect to unactivated output
            m: number of examples
            h_new: height of the output
            w_new: width of the output
            c_new: number of channels in the output
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
                containing the output of the previous layer
            h_prev: height of the previous layer
            w_prev: width of the previous layer
            c_prev: number of channels in the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
            containing the kernels for the convolution
            kh: filter height
            kw: filter width
        padding: string that is either same or valid
            indicating the type of padding used
        stride: tuple of (sh, sw) containing the strides
            sh: stride for the height
            sw: stride for the width
    Returns:
        The gradients with respect to previous layer, kernels, and biases
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'same':
        ph = ((h_new - 1) * sh + kh - h_prev) // 2
        pw = ((w_new - 1) * sw + kw - w_prev) // 2
    elif padding == 'valid':
        ph = pw = 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    dA_prev_pad = np.pad(dA_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        dz = dZ[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dz[h, w, c]
                    dW[:, :, :, c] += a_slice * dz[h, w, c]

    if ph == 0 and pw == 0:
        dA_prev = dA_prev_pad
    else:
        dA_prev = dA_prev_pad[:, ph:-ph if ph != 0 else None, pw:-pw if pw != 0 else None, :]

    return dA_prev, dW, db
