#!/usr/bin/env python3
"""Module for loading the FrozenLake environment"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from gymnasium

    Parameters:
    desc: either None or a list of lists containing a custom description
          of the map to load for the environment
    map_name: either None or a string containing the pre-made map to load
    is_slippery: a boolean to determine if the ice is slippery

    Returns:
    the environment
    """
    return gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi"
    )
