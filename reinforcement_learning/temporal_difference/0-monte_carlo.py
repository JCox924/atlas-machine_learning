#!/usr/bin/env python3
"""performs the Monte Carlo algorithm"""

import numpy as np


def monte_carlo(env, V, policy,
                episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    env = environment instance
        tested with frozenlake
    V = Value Estimate
        np.ndarray
            shape (s,)
    policy = policy function
        function
            recieves state
                returns next action from state
    episodes = number of episodes to train over
        int
    max_steps = steps per episode
    alpha = learning rate
    gamma = discount rate

    Returns:
        V = value estimate (upated)
    """
    num_states = env.observation_space.n
    returns = {State: [] for State in range(num_states)}

    for episode in range(episodes):
        state = env.reset()
        episode_data = []
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminal, _ = env.step(action)
            episode_data.append((state, reward))
            state = next_state
            if terminal:
                break

        G = 0
        for state_g, reward in reversed(episode_data):
            G = gamma * G + reward

            returns[state_g].append(G)
            V[state_g] += alpha * np.mean(returns[state_g])

    return V
