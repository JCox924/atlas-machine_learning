#!/usr/bin/env python3
"""
policy_gradient.py

Defines a simple softmax policy function.
"""

import numpy as np


def policy(matrix, weight):
    """
    Computes action probabilities from inputs via a softmax over
    the linear scores.

    Args:
        matrix (ndarray of shape (m, n)):
            Each row is an input feature vector (e.g., state).
        weight (ndarray of shape (n, k)):
            Weight matrix mapping features to k scores.

    Returns:
        ndarray of shape (m, k):
        Row-wise softmax probabilities over the k actions.
    """
    scores = np.dot(matrix, weight)
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
