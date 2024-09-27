# RLWireless

This repository contains code for implementing reinforcement learning-based resource allocation algorithms for wireless networks. The code is designed to optimize resource allocation in a dynamic wireless environment with changing network conditions and user demands.

## Installation

To install the necessary dependencies and set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/muhd-umer/rl-wireless.git
    cd rl-wireless
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Environment

The environment for this project is based on the `Gymnasium` library; a standard for creating RL environments.<br>
The main environment class is `MassiveMIMOEnv`, which is defined in the `network/environment.py` file.

To register and create the environment, you can use the following code snippet:

```python
import gymnasium as gym
from network import MassiveMIMOEnv

# Set the parameters
N = 7
M = 32
K = 10
Ns = 10
min_P = -20
max_P = 23
num_P = 10
dtype = np.float32

# Register and create the environment
gym.register(id="MassiveMIMO-v0", entry_point=MassiveMIMOEnv)

env = gym.make(   # refer to the implementation for information on what arguments represent
    "MassiveMIMO-v0",
    N=N,
    M=M,
    K=K,
    Ns=Ns,
    min_P=min_P,
    max_P=max_P,
    num_P=num_P,
    dtype=dtype,
)
```

## Usage

To run the code and train an agent of your own, you can use the provided notebook `run.ipynb`, which contains step-by-step instructions and examples.<br>
Below is an example of how to set up and run the environment:

```python
import numpy as np
import gymnasium as gym
from network import MassiveMIMOEnv

# Create the environment
env = gym.make("MassiveMIMO-v0", ...)   # input correct arguments

# Example usage
state = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        state = env.reset()
    else:
        state = next_state
```

Contributing
Contributions are always welcome and highly appreciated. 💖
