import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for value function estimation.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action to take
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: the updated value estimate
    """
    # Make a copy of V to avoid modifying the original
    V = V.copy()

    for episode in range(episodes):
        # Generate an episode
        states = []
        rewards = []

        # Reset environment and get initial state
        state, _ = env.reset()

        # Run episode
        for step in range(max_steps):
            states.append(state)

            # Get action from policy
            action = policy(state)

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)

            # Update state
            state = next_state

            # Check if episode is done
            if terminated or truncated:
                break

        # Calculate returns and update value function
        G = 0

        # Process episode backwards to calculate returns
        for t in reversed(range(len(states))):
            G = gamma * G + rewards[t]

            # Update value function using incremental mean
            state_t = states[t]
            V[state_t] = V[state_t] + alpha * (G - V[state_t])

    return V
