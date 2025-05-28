#!/usr/bin/env python3
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Epsilon-greedy policy"""
    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])


def sarsa_lambtha(env,
                  Q,
                  lambtha,
                  episodes=5000,
                  max_steps=100,
                  alpha=0.1,
                  gamma=0.99,
                  epsilon=1,
                  min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Performs SARSA(lambda) learning to estimate the action-value function.

    Parameters:
    - env: OpenAI Gym environment instance.
    - Q: numpy.ndarray of shape (S, A) containing the Q-table.
    - lambtha: eligibility trace decay factor (lambda).
    - episodes: total number of episodes to run (default 5000).
    - max_steps: maximum steps per episode (default 100).
    - alpha: learning rate (default 0.1).
    - gamma: discount factor (default 0.99).
    - epsilon: initial epsilon for epsilon-greedy policy (default 1).
    - min_epsilon: minimum epsilon after decay (default 0.1).
    - epsilon_decay: amount to subtract from epsilon each episode (default 0.05).

    Returns:
    - Q: The updated Q-table.
    """
    for episode in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)

        E = np.zeros(Q.shape)

        for _ in range(max_steps):
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            E[state, action] = 1

            Q += alpha * delta * E
            E *= gamma * lambtha

            state, action = next_state, next_action

            if done or truncated:
                break

        epsilon = max(min_epsilon, epsilon - np.exp(-epsilon_decay * episode))

    return Q
