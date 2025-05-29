#!/usr/bin/env python3
"""
train.py

Implements a full Monte-Carlo policy gradient training loop (REINFORCE).
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Train a policy using Monte-Carlo policy gradients.

    Args:
        env (gymnasium.Env): the Gym environment instance.
        nb_episodes (int): number of episodes to train over.
        alpha (float): learning rate.
        gamma (float): discount (return) factor.

    Returns:
        list of float: the total reward (score) obtained in each episode.
    """
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # initialize weight matrix (state_dim × action_dim)
    weight = np.random.rand(n_states, n_actions)

    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        grads = []
        rewards = []
        score = 0.0
        done = False

        if show_result and episode % 1000 == 0 and episode > 0:
            env.render()

        while not done:
            action, grad = policy_gradient(state, weight)
            next_state, reward, done, _, _ = env.step(action)

            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = next_state

        returns = np.zeros_like(rewards, dtype=float)
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            returns[t] = G

        # update weights via gradient ascent
        for grad, Gt in zip(grads, returns):
            weight += alpha * grad * Gt

        scores.append(score)
        print(f"Episode: {episode} Score: {score}")

    return scores
