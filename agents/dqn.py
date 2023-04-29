"""
Implementation of DQN agent
"""

import time
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard.writer import SummaryWriter
import tensorboard


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration

    return max(slope * t + start_e, end_e)


class DQNAgent(object):
    """
    A Deep Q-Network (DQN) agent.

    Args:
        envs (gym.Env): The environment to interact with.
        gamma (float): The discount factor.
        tau (float): The soft update coefficient for the target network.
        target_freq (int): The frequency (in steps) with which to update the target network.
        start_e (float): The initial exploration probability.
        end_e (float): The final exploration probability.
        exploration_fraction (float): The fraction of steps to explore.
        learning_starts (int): The number of steps to take before starting to train.
        total_timesteps (int): The total number of steps to train for.
        train_freq (int): The frequency (in steps) with which to train the DQN.
        batch_size (int): The batch size for training the DQN.
        device (str): The device to train the DQN on.
        learning_rate (float): The learning rate for training the DQN.
        buffer_size (int): The size of the replay buffer.
    """

    def __init__(
        self,
        envs: gym.Env,
        gamma=0.1,
        tau=1.0,
        target_freq=500,
        start_e=0.2,
        end_e=0.0001,
        exploration_fraction=0.5,
        learning_starts=1000,
        total_timesteps=500000,
        train_freq=10,
        batch_size=128,
        device="cuda",
        learning_rate=2.5e-4,
        buffer_size=50000,
    ):
        self.envs = envs
        self.device = device

        # Set the hyperparameters.
        self.gamma = gamma
        self.tau = tau
        self.target_freq = target_freq
        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.total_timesteps = total_timesteps

        # Initialize the Q network.
        self.q_network = QNetwork(envs).to(self.device)

        # Initialize the optimizer.
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Initialize the target network.
        self.target_network = QNetwork(envs).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Initialize the replay buffer.
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
            handle_timeout_termination=False,
        )

        self.obs, _ = self.envs.reset()
        self.actions = None
        self.writer = SummaryWriter(f"runs/dqn")

    def load_model(self, weight_path):
        """
        Load the model from a given path.
        Args:
            weight_path (str): The path to the model weights.
        """
        self.q_network.load_state_dict(torch.load(weight_path))

    def update_target_network(self):
        # For each parameter in the target network, copy the corresponding parameter
        # from the Q network with a certain learning rate.
        for target_network_param, q_network_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_network_param.data.copy_(
                self.tau * q_network_param.data
                + (1.0 - self.tau) * target_network_param.data
            )

    def get_actions(self, global_step):
        """
        Get the actions from the policy.

        Args:
            global_step (int): The current global step.
        """
        epsilon = linear_schedule(
            self.start_e,
            self.end_e,
            self.exploration_fraction * self.total_timesteps,
            global_step,
        )

        # If the exploration probability is greater than a random number, take a random action.
        if random.random() < epsilon:
            self.actions = np.array(
                [
                    self.envs.single_action_space.sample()
                    for _ in range(self.envs.num_envs)
                ]
            )

        # Otherwise, take the action with the highest Q value.
        else:
            q_values = self.q_network(torch.Tensor(self.obs).to(self.device))
            self.actions = torch.argmax(q_values, dim=1).cpu().numpy()

    def predict_power(self):
        """
        Predict the power of the next action.

        Returns:
            power (float): The predicted power of the next action.
        """
        actions = self.get_actions(self.obs)
        power = self.envs.single_action_space[actions]

        return power

    def train(self, global_step, start_time):
        """
        Train the agent.

        Args:
            global_step (int): The current global step.
            start_time (float): The start time of the training.
            save_model (bool): Whether to save the model after training.

        Returns:
            None.
        """
        if global_step > self.learning_starts:
            if global_step % self.train_freq == 0:
                data = self.replay_buffer.sample(self.batch_size)
                with torch.no_grad():
                    target_max, _ = self.target_network(data.next_observations).max(
                        dim=1
                    )
                    td_target = data.rewards.flatten() + self.gamma * target_max * (
                        1 - data.dones.flatten()
                    )
                old_val = (
                    self.q_network(data.observations).gather(1, data.actions).squeeze()
                )
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    self.writer.add_scalar("losses/td_loss", loss, global_step)
                    self.writer.add_scalar(
                        "charts/epsilon",
                        linear_schedule(
                            self.start_e,
                            self.end_e,
                            self.exploration_fraction * self.total_timesteps,
                            global_step,
                        ),
                        global_step,
                    )
                    self.writer.add_scalar(
                        "charts/reward", data.rewards.mean().item(), global_step
                    )
                    self.writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

                # optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # update target network
            if global_step % self.target_freq == 0:
                self.update_target_network()
