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


def policy_gradient(state, weight):
    """
    Samples an action from the policy for a single state and
    returns both the action index and the gradient of the
    log‐probability ∇_w log π(a|s).

    Args:
        state (ndarray of shape (n,)):
            Feature vector for the current observation.
        weight (ndarray of shape (n, k)):
            Weight matrix mapping features to k action scores.

    Returns:
        action (int): sampled action index.
        grad (ndarray of shape (n, k)):
            Gradient of log π(a|s) with respect to each weight.
            grad[i, j] = state[i] * (I[j == action] - π_j).
    """
    s = np.array(state).reshape(-1)
    probs = policy(s[np.newaxis, :], weight)[0]

    # sample an action (uses np.random, so seeding works)
    action = np.random.choice(probs.shape[0], p=probs)

    one_hot = np.zeros_like(probs)
    one_hot[action] = 1.0

    grad = s[:, None] * (one_hot - probs)[None, :]

    return action, grad
