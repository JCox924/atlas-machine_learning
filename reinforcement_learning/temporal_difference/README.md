# Reinforcement Learning Prediction and Control Algorithms

This repository contains implementations of Monte Carlo prediction, TD(λ) prediction, and SARSA(λ) control algorithms for value function and action-value function estimation in OpenAI Gym environments.

## Files

* `0-monte_carlo.py`: First-visit Monte Carlo prediction to evaluate a policy.
* `1-td_lambtha.py`: TD(λ) prediction with eligibility traces for policy evaluation.
* `2-sarsa_lambtha.py`: SARSA(λ) control with accumulating eligibility traces and ε-greedy action selection.

Each file implements a single function:

```python
V = monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99)

V = td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99)

Q = sarsa_lambtha(env, Q, lambtha, episodes=5000,
                  max_steps=100, alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05)
```

## Requirements

* Python 3.9
* numpy 1.25.2
* gymnasium 0.29.1

Ensure all files are executable and have a shebang line:

```bash
#!/usr/bin/env python3
```

## Installation

```bash
git clone <repo_url>
cd <repo_directory>
pip install numpy gymnasium
chmod +x *.py
```

## Usage

Each algorithm can be tested with the provided `*-main.py` scripts:

```bash
./0-main.py   # Monte Carlo prediction example
./1-main.py   # TD(lambda) prediction example
./2-main.py   # SARSA(lambda) control example
```

These scripts:

1. Create the `FrozenLake8x8-v1` environment.
2. Initialize value (`V`) or action-value (`Q`) arrays.
3. Set a random seed for reproducibility.
4. Run the algorithm and print the learned estimates.

## Coding Standards

* Use `pycodestyle` (v2.11.1) style guidelines.
* All modules, classes, and functions have documentation strings.
* Files end with a newline and are executable.

## License

This project is released under the MIT License.
