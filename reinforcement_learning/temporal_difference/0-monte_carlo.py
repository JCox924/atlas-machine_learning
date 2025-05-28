#!/usr/bin/env python3
"""Monte Carlo Algorithm implementation."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """Performs the Monte Carlo algorithm.

    Args:
        env (gym.Env): Gym environment.
        V (np.ndarray): Value estimate.
        policy (function): Policy function.
        episodes (int): Number of episodes to train.
        max_steps (int): Maximum steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount rate.

    Returns:
        np.ndarray: Updated value estimate.
    """
    for episode in range(episodes):
        state, _ = env.reset()
        episode_data = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, reward))
            state = next_state

            if terminated or truncated:
                break

        G = 0
        for state, reward in reversed(episode_data):
            G = reward + gamma * G
            V[state] = V[state] + alpha * (G - V[state])

    return V
