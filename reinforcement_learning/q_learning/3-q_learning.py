#!/usr/bin/env python3
"""Module for implementing Q-learning algorithm"""
import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning

    Parameters:
    env: the FrozenLakeEnv instance
    Q: a numpy.ndarray containing the Q-table
    episodes: the total number of episodes to train over
    max_steps: the maximum number of steps per episode
    alpha: the learning rate
    gamma: the discount rate
    epsilon: the initial threshold for epsilon greedy
    min_epsilon: the minimum value that epsilon should decay to
    epsilon_decay: the decay rate for updating epsilon between episodes

    Returns:
    Q, total_rewards
    - Q is the updated Q-table
    - total_rewards is a list containing the rewards per episode
    """
    epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            next_state, reward, done, truncated, _ = env.step(action)

            if done and reward == 0:
                reward = -1

            Q[state, action] = Q[state, action] + alpha * (
                    reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        total_rewards.append(total_reward)

        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q, total_rewards
