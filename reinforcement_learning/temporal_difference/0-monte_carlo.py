#!/usr/bin/env python3
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate state values.

    Args:
        env: Gym environment instance.
        V (numpy.ndarray): initial state value estimates.
        policy: function that maps a state to an action.
        episodes (int): number of episodes to simulate.
        max_steps (int): maximum steps per episode.
        alpha (float): learning rate.
        gamma (float): discount factor.

    Returns:
        numpy.ndarray: updated state value estimates.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, done, truncated, _ = env.step(action)
            episode.append((state, reward))
            if done or truncated:
                break
            state = new_state

        G = 0
        for state, reward in reversed(episode):
            G = reward + gamma * G
            V[state] = V[state] + alpha * (G - V[state])

    return V
