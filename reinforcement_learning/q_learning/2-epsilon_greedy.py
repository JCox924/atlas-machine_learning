#!/usr/bin/env python3
"""Module for implementing the epsilon-greedy algorithm"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action

    Parameters:
    Q: a numpy.ndarray containing the q-table
    state: the current state
    epsilon: the epsilon to use for the calculation

    Returns:
    the next action index
    """
    # Determine if we should explore or exploit
    p = np.random.uniform(0, 1)

    # Explore (random action)
    if p < epsilon:
        action = np.random.randint(0, Q.shape[1])
    # Exploit (best action)
    else:
        action = np.argmax(Q[state])

    return action
