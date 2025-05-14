#!/usr/bin/env python3
"""Module for initializing the Q-table"""
import numpy as np


def q_init(env):
    """
    Initializes the Q-table

    Parameters:
    env: the FrozenLakeEnv instance

    Returns:
    the Q-table as a numpy.ndarray of zeros
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))

    return Q
