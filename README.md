# RLWireless

This repository contains code for implementing reinforcement learning-based resource allocation algorithms for wireless networks. The code is designed to optimize resource allocation in a dynamic wireless environment with changing network conditions and user demands.

The application of deep reinforcement learning (DRL) for dynamic resource allocation in wireless communication systems is explored in this project. An environment simulates a multi-cell massive MIMO wireless system. DRL algorithms such as DQN and PPO are used to optimize resource allocation, demonstrating improved efficiency over traditional methods. For more details, refer to the [report](./report/report.pdf).

## Installation

To install the necessary dependencies and set up the project, follow these steps:

### Clone the repository

```sh
git clone https://github.com/muhd-umer/rl-wireless.git
cd rl-wireless
```

### Create a new virtual environment

It is recommended to create a new virtual environment to avoid conflicts with other projects. You can create a new virtual environment using `conda` or `mamba`:

**→ Using Conda**

```sh
conda create -n rl-wireless python=3.9
conda activate rl-wireless
pip install -r requirements.txt
```

**→ Using Mamba**

```sh
wget -O miniforge.sh \
     "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash miniforge.sh -b -p "${HOME}/conda"

source "${HOME}/conda/etc/profile.d/conda.sh"
source "${HOME}/conda/etc/profile.d/mamba.sh"

conda activate
mamba create -n rl-wireless python=3.9
mamba activate rl-wireless
pip install -r requirements.txt
```

## Environment

The environment for this project is based on the `Gymnasium` library, a standard for creating RL environments. The main environment class is `MassiveMIMOEnv`, which is defined in the `network/environment.py` file.

### Environment Setup

To register and create the environment, you can use the following code snippet:

```python
import gymnasium as gym
from network import MassiveMIMOEnv
import numpy as np

# Set the parameters
N = 7          # Number of cells (or base stations)
M = 32         # Number of antennas per base station
K = 10         # Number of user equipments (UEs) per cell
Ns = 10        # Number of samples for the channel realization
min_P = -20    # Minimum transmission power in dBm
max_P = 23     # Maximum transmission power in dBm
num_P = 10     # Number of discrete power levels
dtype = np.float32    # Data type for computations

# Register and create the environment
gym.register(id="MassiveMIMO-v0", entry_point=MassiveMIMOEnv)

env = gym.make(
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

To run the code and train an agent, you can use the provided notebook `run.ipynb`, which contains step-by-step instructions and examples. Below is an example of how to set up and run the environment:

```python
import numpy as np
import gymnasium as gym
from network import MassiveMIMOEnv

# Create the environment
env = gym.make("MassiveMIMO-v0", N=7, M=32, K=10, Ns=10, min_P=-20, max_P=23, num_P=10, dtype=np.float32)

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

## Training

The `run.ipynb` notebook demonstrates how to train various DRL agents using the Ray RLlib library.<br>
It trains PPO, DQN, and R2D2 agents and evaluates their performance.

### Example: Training a PPO Agent

```python
import ray
from ray import air, tune
from ray.tune.registry import get_trainable_cls

# Initialize Ray
ray.init()

# Configuration for PPO
config = (
    get_trainable_cls("PPO")
    .get_default_config()
    .environment("MassiveMIMO-v0")
    .framework("torch")
    .resources(num_gpus=0.5)
    .rollouts(num_rollout_workers=1)
    .training(lr=tune.grid_search([0.005, 0.0001]))
)

# Training
results = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop={"timesteps_total": 100000}, local_dir="./results"),
).fit()
```

## Contributing

Contributions are always welcome and highly appreciated. 💖
