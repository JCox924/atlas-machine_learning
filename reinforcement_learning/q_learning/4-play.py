#!/usr/bin/env python3
"""Module for playing the Frozen Lake game with a trained agent"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode

    Parameters:
    env: the FrozenLakeEnv instance
    Q: a numpy.ndarray containing the Q-table
    max_steps: the maximum number of steps in the episode

    Returns:
    total_rewards, rendered_outputs - total rewards for the episode and
    a list of rendered outputs representing the board state at each step
    """
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = []

    for step in range(max_steps):
        rendered = env.render()

        action = np.argmax(Q[state])

        action_names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
        rendered += f"\n  ({action_names[action]})"
        rendered_outputs.append(rendered)

        next_state, reward, done, truncated, _ = env.step(action)

        state = next_state
        total_rewards += reward

        if done or truncated:
            final_rendered = env.render()
            rendered_outputs.append(final_rendered)
            break

    return total_rewards, rendered_outputs
