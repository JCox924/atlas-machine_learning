#!/usr/bin/env python3
"""
Module for performing back propagation over a convolutional layer
"""
import numpy as np


def pad_valid(A_prev):
    """
    Performs valid padding (no padding)
    Args:
        A_prev: Input to be padded
    Returns:
        Padded input (same as input for valid padding)
    """
    return A_prev


def pad_same(A_prev, kh, kw, sh, sw):
    """
    Performs same padding on images
    Args:
        A_prev: Input to be padded
        kh: Filter height
        kw: Filter width
        sh: Stride height
        sw: Stride width
    Returns:
        Padded input
    """
    ph = max((((A_prev.shape[1] - 1) * sh + kh - A_prev.shape[1]) // 2), 0)
    pw = max((((A_prev.shape[2] - 1) * sw + kw - A_prev.shape[2]) // 2), 0)

    return np.pad(A_prev,
                  ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                  mode='constant')


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
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
        b: numpy.ndarray of shape (1, 1, 1, c_new)
            containing the biases
        padding: string that is either same or valid
            indicating the type of padding used
        stride: tuple of (sh, sw) containing the strides
            sh: stride for the height
            sw: stride for the width
    Returns:
        The gradients with respect to previous layer, kernels, and biases
    """
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kh = W.shape[0]
    kw = W.shape[1]
    c_new = W.shape[3]

    h_new = dZ.shape[1]
    w_new = dZ.shape[2]

    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == "valid":
        padded = pad_valid(A_prev)
        ph, pw = 0, 0
    elif padding == "same":
        padded = pad_same(A_prev, kh, kw, sh, sw)
        ph = max((((h_prev - 1) * sh + kh - h_prev) // 2), 0)
        pw = max((((w_prev - 1) * sw + kw - w_prev) // 2), 0)
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    dA_padded = np.zeros_like(padded)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    a_slice = padded[i, h_start:h_end, w_start:w_end, :]

                    dA_padded[i, h_start:h_end, w_start:w_end, :] += (
                            W[:, :, :, c] * dZ[i, h, w, c]
                    )
                    dW[:, :, :, c] += (
                            a_slice * dZ[i, h, w, c]
                    )

    if padding == 'valid':
        dA_prev = dA_padded
    else:
        dA_prev = dA_padded[:, ph:h_prev + ph, pw:w_prev + pw, :]

    return dA_prev, dW, db
