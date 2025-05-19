#!/usr/bin/env python3
"""
Script to train a DQN agent to play Atari's Breakout
"""
import gymnasium as gym
import numpy as np
from tensorflow.keras import mixed_precision
import tensorflow as tf
# Use tf.keras for models/optimizers and ensure h5py-backed saving
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers.legacy import Adam

# Monkey-patch keras version for keras-rl2 compatibility
import tensorflow.keras as keras
keras.__version__ = tf.__version__

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.processors import Processor
from PIL import Image
mixed_precision.set_global_policy("mixed_float16")
# Enable GPU memory growth if a GPU is present
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
print('GPUs found:', gpus)

class AtariProcessor(Processor):
    """
    Preprocessor for Atari frames: grayscale + resize + reward clipping.
    """
    def process_observation(self, observation):
        assert observation.ndim == 3  # (H, W, C)
        img = Image.fromarray(observation).convert('L').resize((84, 84))
        return np.array(img, dtype=np.uint8)

    def process_state_batch(self, batch):
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)


class GymCompatibilityWrapper(gym.Wrapper):
    """
    Adapts Gymnasium env API to keras-rl2 expectations.
    """
    def reset(self, **kwargs):
        state, _ = self.env.reset(**kwargs)
        return state

    def step(self, action):
        ns, reward, term, trunc, info = self.env.step(action)
        done = term or trunc
        return ns, reward, done, info

    def render(self, *args, **kwargs):
        return self.env.render()


def build_model(height, width, channels, actions):
    """
    Creates a CNN (channels_first) matching DeepMind-style DQN.
    """
    model = Sequential([
        Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
               input_shape=(channels, height, width), data_format='channels_first'),
        Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(actions, activation='linear')
    ])
    return model


def build_agent(model, actions):
    """
    Configures and compiles the DQNAgent with replay and epsilon-greedy policy.
    """
    memory = SequentialMemory(limit=1_000_000, window_length=4)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), attr='eps',
        value_max=1.0, value_min=0.1, value_test=0.05,
        nb_steps=1_000_000
    )
    processor = AtariProcessor()

    agent = DQNAgent(
        model=model,
        nb_actions=actions,
        memory=memory,
        nb_steps_warmup=50_000,
        target_model_update=10_000,
        policy=policy,
        processor=processor,
        enable_double_dqn=True,
        enable_dueling_network=False
    )
    agent.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    return agent


def main():
    env_name = 'ALE/Breakout-v5'
    env = gym.make(env_name, render_mode=None)
    env = GymCompatibilityWrapper(env)

    height, width, channels = 84, 84, 4
    actions = env.action_space.n

    model = build_model(height, width, channels, actions)
    agent = build_agent(model, actions)

    # Train for 1e6 steps (adjust frame count), log every 10k
    agent.fit(env, nb_steps=1_000_000, visualize=False, verbose=1, log_interval=10_000)

    # Save trained policy network
    agent.save_weights('policy.h5', overwrite=True)
    env.close()


if __name__ == '__main__':
    main()
