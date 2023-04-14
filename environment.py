"""
Implementation of Multi-Cell Massive MIMO Environment
"""
import jax
from jax import jit, vmap
import jax.numpy as jnp
import scipy as sp
import numpy as np
import pandas as pd
import scipy.io as sc
from scipy import special
from scipy.constants import *
from scipy.special import erfinv
from scipy.integrate import quad
from scipy.linalg import toeplitz
import gymnasium as gym
from gymnasium import Env, error, spaces, utils
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import seeding

dtype = jnp.float32


class MassiveMIMOEnv(Env):
    """
    Multi-Cell Massive MIMO Environment
    """

    def __init__(self, N, M, K, BW, NF, Ns, min_P, max_P, num_P):
        """
        Initialize the environment
        """
        self.N = N  # number of cells is equals to number of BSs
        self.M = M  # number of BS transmission antnneas
        self.K = K  # number of UEs in a cell
        self.BW = BW  # Bandwidth = 10MHz
        self.NF = NF  # Power of noise figure [dBm]
        self.Ns = Ns  # Number of sample
        self.min_P = min_P  # Minimum transmission power [dBm]
        self.max_P = max_P  # Maximum transmission power [dBm]
        self.num_P = num_P  # Number of action space

        # Channel parameters and matrices
        self.precoding_matrix = self.antenna_selection()
        self.H, self.H_gain = self.get_channel_mats(no_realization=self.Ns)

        self.power_alloc = self.get_power_alloc()  # Power allocation (action value)
        self.num_actions = len(self.power_alloc)  # Number of actions
        self.action_space = spaces.Discrete(self.num_actions)  # Action space
        mem_limits = np.finfo(jnp.float32).max  # Maximum value of float32

        # Observation space: contains transmission power, SINR, and sum rate
        self.observation_space = spaces.Box(
            low=-mem_limits, high=mem_limits, dtype=jnp.float32
        )  # Observation space

        # Initialize the state
        self.state = None
        self.downlink_rate = None
        self.rate_list = []

    """
    Downlink Scenario
    """

    @jit
    def get_power_alloc(self):
        """
        Get power allocation
        """
        power_alloc = jnp.hstack(
            [
                jnp.zeros((1), dtype=dtype),
                1e-3
                * pow(
                    10.0, jnp.linspace(self.min_P, self.max_P, self.num_P - 1) / 10.0
                ),
            ]
        )
        return power_alloc

    @jit
    def hermitian(self, A):
        """
        Hermitian of a matrix
        """
        return jnp.transpose(jnp.conj(A))
