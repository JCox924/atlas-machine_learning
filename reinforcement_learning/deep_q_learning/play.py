# Directory: reinforcement_learning/deep_q_learning
# File: play.py
#!/usr/bin/env python3
"""
play.py: Load a trained DQN policy and play Breakout using GreedyQPolicy.
"""
import gymnasium as gym
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
from rl.processors import Processor
from train import build_model, GymCompatibilityWrapper
import numpy as np


class AtariProcessor(Processor):
    """
    Preprocessor for Atari frames: grayscale+resize+normalize to [0,1].
    """
    def process_observation(self, observation):
        # observation: raw RGB frame (210,160,3)
        from PIL import Image
        img = Image.fromarray(observation).convert('L').resize((84, 84))
        frame = np.array(img, dtype='uint8')
        return frame

    def process_state_batch(self, batch):
        # batch: array of stacked frames (batch_size, window_length, H, W)
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)


def main():
    # Create raw Breakout env
    env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
    # Wrap for Keras-RL2 compatibility
    env = GymCompatibilityWrapper(env)

    # Environment properties
    height, width, channels = 84, 84, 4
    nb_actions = env.action_space.n  # 4 actions

    # Build model matching train.py signature
    model = build_model(height, width, channels, nb_actions)

    # Setup memory, policy, and processor
    memory = SequentialMemory(limit=1_000_000, window_length=4)
    policy = GreedyQPolicy()
    processor = AtariProcessor()

    # Instantiate DQNAgent
    agent = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        policy=policy,
        processor=processor,
        nb_steps_warmup=50_000,
        target_model_update=10_000,
        enable_double_dqn=True,
        enable_dueling_network=False,
        test_policy=policy
    )
    agent.compile('adam')

    # Load trained weights
    agent.load_weights('policy.h5')

    # Test the agent for 5 episodes
    history = agent.test(env, nb_episodes=5)
    scores = history.history.get('episode_reward', [])
    print(f"Average Score over 5 episodes: {np.mean(scores):.2f}")

    env.close()


if __name__ == '__main__':
    main()
