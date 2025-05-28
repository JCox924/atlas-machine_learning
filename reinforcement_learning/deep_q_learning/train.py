#!/usr/bin/env python3
"""
train.py: Train a DQN agent to play Atari's Breakout with performance optimizations.
"""
import os
import gymnasium as gym
import numpy as np
import tensorflow as tf

# GPU memory growth
from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy("mixed_float16")  # disabled for cudnn stability

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
print('GPUs found:', gpus)

# Control threading to avoid oversubscription
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Use tf.keras for models/optimizers and ensure h5py-backed saving
import tensorflow.keras as keras
keras.__version__ = tf.__version__
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers.legacy import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.processors import Processor

class AtariProcessor(Processor):
    """
    Preprocessor: no-op resizing (handled by wrapper) + reward clipping.
    """
    def process_observation(self, observation):
        # observation is 84x84 uint8 from AtariPreprocessing
        return observation

    def process_state_batch(self, batch):
        # Normalize to [0,1] float32
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        # Clip rewards
        return np.clip(reward, -1.0, 1.0)

class GymCompatibilityWrapper(gym.Wrapper):
    """
    Adapts Gymnasium to keras-rl2 API.
    """
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        return result[0] if isinstance(result, tuple) else result

    def step(self, action):
        ns, reward, term, trunc, info = self.env.step(action)
        done = term or trunc
        return ns, reward, done, info

    def render(self, *args, **kwargs):
        return self.env.render()


def build_model(height, width, channels, actions):
    """
    Creates a CNN (channels_first) matching DeepMind's DQN.
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
    Configures and compiles the DQNAgent.
    """
    memory = SequentialMemory(limit=1_000_000, window_length=4)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), attr='eps',
        value_max=1.0, value_min=0.1, value_test=0.05,
        nb_steps=500_000
    )
    processor = AtariProcessor()

    agent = DQNAgent(
        model=model,
        nb_actions=actions,
        memory=memory,
        policy=policy,
        processor=processor,
        nb_steps_warmup=50_000,
        target_model_update=10_000,
        enable_double_dqn=True,
        enable_dueling_network=True,
        train_interval=4,
        batch_size=64,
        gamma=0.99,
        delta_clip=1.0
    )
    agent.compile(Adam(learning_rate=0.0005), metrics=['mae'])
    return agent


def main():
    from gymnasium.wrappers import AtariPreprocessing

    # Single env: preprocessing only, memory will stack frames
    env = gym.make('BreakoutNoFrameskip-v4', render_mode=None)
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=False  # keep uint8 for processor
    )
    env = GymCompatibilityWrapper(env)

    height, width, channels = 84, 84, 4
    actions = gym.make('BreakoutNoFrameskip-v4').action_space.n

    model = build_model(height, width, channels, actions)
    agent = build_agent(model, actions)

    # Train for 1M steps, logging every 10k
    agent.fit(env, nb_steps=1_000_000, visualize=False, verbose=2, log_interval=10_000)

    agent.save_weights('policy.h5', overwrite=True)
    env.close()


if __name__ == '__main__':
    main()