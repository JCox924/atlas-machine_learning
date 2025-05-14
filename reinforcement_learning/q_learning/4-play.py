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

    def custom_render(state_idx, env_desc, action=None):
        """Custom render function to create consistent output format"""
        height = len(env_desc)
        width = len(env_desc[0])
        output = ""

        for i in range(height):
            for j in range(width):
                idx = i * width + j
                if idx == state_idx:
                    output += f'"{env_desc[i][j].decode("utf-8")}"'
                else:
                    output += env_desc[i][j].decode("utf-8")
            if i < height - 1:
                output += "\n"

        if action is not None:
            output += f"\n  ({action})"

        return output

    action_names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
    env_desc = env.unwrapped.desc

    for step in range(max_steps):
        action = np.argmax(Q[state])

        render = custom_render(state, env_desc, action_names[action])
        rendered_outputs.append(render)

        next_state, reward, done, truncated, _ = env.step(action)

        state = next_state
        total_rewards += reward

        if done or truncated:
            final_render = custom_render(state, env_desc)
            rendered_outputs.append(final_render)
            break

    return total_rewards, rendered_outputs
