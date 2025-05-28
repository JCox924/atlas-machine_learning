#!/usr/bin/env python3
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm for value function estimation.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action to take
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: the updated value estimate
    """
    V = V.copy()

    for episode in range(episodes):
        eligibility_traces = np.zeros(len(V))

        state, _ = env.reset()

        for step in range(max_steps):
            action = policy(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            td_error = reward + gamma * V[next_state] - V[state]

            eligibility_traces[state] += 1.0

            V += alpha * td_error * eligibility_traces

            eligibility_traces *= gamma * lambtha

            state = next_state

            if terminated or truncated:
                break

    return V
