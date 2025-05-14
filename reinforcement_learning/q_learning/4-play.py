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
        # Render the current state
        rendered = env.render()
        rendered_outputs.append(rendered)

        # Choose the best action according to the Q-table (exploit)
        action = np.argmax(Q[state])

        # Add the action to the output
        action_names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
        rendered_outputs[-1] += f"\n  ({action_names[action]})"

        # Take action
        next_state, reward, done, truncated, _ = env.step(action)

        # Update state and reward
        state = next_state
        total_rewards += reward

        if done or truncated:
            # Render final state
            final_rendered = env.render()
            rendered_outputs.append(final_rendered)
            break

    return total_rewards, rendered_outputs