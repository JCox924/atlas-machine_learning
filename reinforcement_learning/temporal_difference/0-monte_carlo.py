#!/usr/bin/env python3
"""
0-monte_carlo.py: Implementation of Monte Carlo prediction for value function estimation.
"""
import numpy as np

def monte_carlo(env, V, policy,
                 episodes=5000,
                 max_steps=100,
                 alpha=0.1,
                 gamma=0.99):
    """
    Performs first-visit Monte Carlo prediction to evaluate a given policy.

    Parameters:
    - env: The OpenAI Gym environment instance.
    - V: numpy.ndarray of shape (S,) containing the current value estimates for each state.
    - policy: A function that takes a state and returns an action.
    - episodes: Total number of episodes to sample (default 5000).
    - max_steps: Maximum steps per episode (default 100).
    - alpha: Learning rate (default 0.1).
    - gamma: Discount factor (default 0.99).

    Returns:
    - V: The updated value function estimates.
    """
    for _ in range(episodes):
        states = []
        rewards = []
        state, _ = env.reset()
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            states.append(state)
            rewards.append(reward)
            state = next_state
            if terminated or truncated:
                break

        G = 0.0
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            s_t = states[t]
            if s_t not in states[:t]:
                V[s_t] += alpha * (G - V[s_t])

    return V
