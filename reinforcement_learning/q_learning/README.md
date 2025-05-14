# Q-Learning Implementation

This directory contains a complete implementation of the Q-Learning algorithm, a fundamental reinforcement learning technique. The implementation is broken down into several modular Python scripts for better understanding and clarity.

## Files Structure

- `0-load_env.py`: Environment loader and setup
- `1-q_init.py`: Q-table initialization
- `2-epsilon_greedy.py`: Implementation of the epsilon-greedy policy for action selection
- `3-q_learning.py`: Core Q-learning algorithm implementation
- `4-play.py`: Script to run and visualize the trained agent

## Overview

Q-Learning is a model-free reinforcement learning algorithm that learns to make optimal decisions by learning an action-value function (Q-function). This implementation follows the standard Q-learning process:

1. Environment setup and initialization
2. Q-table initialization
3. Action selection using epsilon-greedy strategy
4. Learning through interaction with the environment
5. Demonstration of learned behavior

## Getting Started

### Prerequisites

- Python 3.x
- Required packages (install via pip):
  ```bash
  pip install numpy gym
  ```

### Usage

The scripts should be run in sequential order:

1. First, set up the environment:
   ```bash
   python 0-load_env.py
   ```

2. Initialize the Q-table:
   ```bash
   python 1-q_init.py
   ```

3. Train the agent using Q-learning:
   ```bash
   python 3-q_learning.py
   ```

4. Watch the trained agent play:
   ```bash
   python 4-play.py
   ```

## Key Components

- **Environment Loading**: Sets up the OpenAI Gym environment for training
- **Q-Table Initialization**: Creates and initializes the Q-table with zeros or random values
- **Epsilon-Greedy Strategy**: Balances exploration and exploitation during training
- **Q-Learning Algorithm**: Implements the core learning algorithm with state-action value updates
- **Visualization**: Allows viewing the trained agent's performance

## Algorithm Overview

Q-Learning updates the Q-values using the formula:

Q(s,a) = Q(s,a) + α[R + γ max(Q(s',a')) - Q(s,a)]

Where:
- s, a are the current state and action
- s' is the next state
- α is the learning rate
- γ is the discount factor
- R is the reward

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.