#!/usr/bin/env python3
"""
Module for performing back propagation over a convolutional layer
"""
import numpy as np


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
    h_prev, w_prev = A_prev.shape[1], A_prev.shape[2]

    out_h = int(np.ceil(float(h_prev) / float(sh)))
    out_w = int(np.ceil(float(w_prev) / float(sw)))

    pad_h = max((out_h - 1) * sh + kh - h_prev, 0)
    pad_w = max((out_w - 1) * sw + kw - w_prev, 0)

    ph_top = pad_h // 2
    ph_bottom = pad_h - ph_top
    pw_left = pad_w // 2
    pw_right = pad_w - pw_left

    padded = np.pad(A_prev,
                    ((0, 0),
                     (ph_top, ph_bottom),
                     (pw_left, pw_right),
                     (0, 0)),
                    mode='constant')
    return padded, ph_top, ph_bottom, pw_left, pw_right


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
    m, h_prev, w_prev, c_prev = A_prev.shape

    kh, kw, _, c_new = W.shape

    _, h_new, w_new, _ = dZ.shape

    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == "valid":
        padded = A_prev
        ph_top, ph_bottom, pw_left, pw_right = 0, 0, 0, 0
    elif padding == "same":
        padded, ph_top, ph_bottom, pw_left, pw_right = pad_same(A_prev, kh, kw, sh, sw)

    else:
        raise ValueError("padding must be 'same' or 'valid'")

    dA_padded = np.zeros_like(padded)

    for i in range(m):
        A_prev_padded = padded[i]
        dA_prev_padded = dA_padded[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    a_slice = A_prev_padded[h_start:h_end, w_start:w_end, :]

                    dA_prev_padded[h_start:h_end, w_start:w_end, :] += (
                            W[:, :, :, c] * dZ[i, h, w, c]
                    )
                    dW[:, :, :, c] += (
                            a_slice * dZ[i, h, w, c]
                    )

    if padding == 'valid':
        dA_prev = dA_padded
    else:
        dA_prev = (
                      dA_padded)[:, ph_top:ph_top + h_prev, pw_left:pw_left + w_prev, :]

    return dA_prev, dW, db
