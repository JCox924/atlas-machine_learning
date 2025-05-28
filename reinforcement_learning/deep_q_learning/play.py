#!/usr/bin/env python3
"""
play.py: Load a trained DQN policy and play Breakout using GreedyQPolicy.
"""
import gymnasium as gym
import numpy as np

from train import build_model, GymCompatibilityWrapper, AtariProcessor
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.optimizers.legacy import Adam


def main():
    # Create and wrap the environment exactly as in train.py
    env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=False  # keep uint8 frames for the processor
    )
    env = GymCompatibilityWrapper(env)

    # Environment properties
    height, width, channels = 84, 84, 4
    nb_actions = env.action_space.n  # should be 4

    # Build model matching train.py
    model = build_model(height, width, channels, nb_actions)

    # Memory, policy, processor
    memory = SequentialMemory(limit=1_000_000, window_length=4)
    policy = GreedyQPolicy()
    processor = AtariProcessor()

    # Instantiate DQNAgent with dueling enabled
    agent = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        policy=policy,
        processor=processor,
        nb_steps_warmup=50_000,
        target_model_update=10_000,
        enable_double_dqn=True,
        enable_dueling_network=True,
        test_policy=policy
    )
    # Compile with same optimizer & learning rate as training
    agent.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    # Load trained weights
    agent.load_weights('policy.h5')

    # Test the agent for 5 episodes
    history = agent.test(env, nb_episodes=5, visualize=True)
    scores = history.history.get('episode_reward', [])
    print(f"Average Score over 5 episodes: {np.mean(scores):.2f}")

    env.close()


if __name__ == '__main__':
    main()
