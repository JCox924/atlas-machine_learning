#!/usr/bin/env python3
import numpy as np

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
    n_states, n_actions = Q.shape
    epsilon_curr = epsilon

    for _ in range(episodes):
        E = np.zeros_like(Q)
        state, _ = env.reset()

        if np.random.uniform() < epsilon_curr:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)

            if np.random.uniform() < epsilon_curr:
                next_action = np.random.randint(n_actions)
            else:
                next_action = np.argmax(Q[next_state])

            Q_next = 0 if (terminated or truncated) else Q[next_state, next_action]

            delta = reward + gamma * Q_next - Q[state, action]

            E[state, action] = 1

            Q += alpha * delta * E

            E *= gamma * lambtha

            state, action = next_state, next_action

            if terminated or truncated:
                break

        epsilon_curr = max(min_epsilon, epsilon_curr - epsilon_decay)

    return Q
