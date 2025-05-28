#!/usr/bin/env python3
"""
1-td_lambtha.py: Implementation of TD(lambda) prediction for value function estimation.
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha,
               episodes=5000,
               max_steps=100,
               alpha=0.1,
               gamma=0.99):
    """
    Performs TD(lambda) prediction to evaluate a given policy.

    Parameters:
    - env: The OpenAI Gym environment instance.
    - V: numpy.ndarray of shape (S,) containing the current value estimates for each state.
    - policy: A function that takes a state and returns an action.
    - lambtha: Eligibility trace decay factor (lambda).
    - episodes: Total number of episodes to sample (default 5000).
    - max_steps: Maximum steps per episode (default 100).
    - alpha: Learning rate (default 0.1).
    - gamma: Discount factor (default 0.99).

    Returns:
    - V: The updated value function estimates.
    """
    n_states = V.shape[0]

    for _ in range(episodes):
        E = np.zeros(n_states)
        state, _ = env.reset()

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            V_next = 0 if (terminated or truncated) else V[next_state]
            delta = reward + gamma * V_next - V[state]
            E[state] += 1
            V += alpha * delta * E
            E *= gamma * lambtha
            state = next_state
            if terminated or truncated:
                break

    return V
