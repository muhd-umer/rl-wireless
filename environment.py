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
from scipy.special import erfinv
from scipy.integrate import quad
from scipy.linalg import toeplitz
import gymnasium as gym
from gymnasium import Env, error, spaces, utils
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import seeding
from functools import partial

dtype = jnp.float32


class MassiveMIMOEnv(Env):
    """
    Multi-Cell Massive MIMO Environment
    """

    def __init__(self, N, M, K, BW, NF, Ns, min_P, max_P, num_P, dtype=dtype):
        """
        Initialize the environment
        """
        self.dtype = dtype
        self.N = N  # number of cells is equals to number of BSs
        self.M = M  # number of BS transmission antnneas
        self.K = K  # number of UEs in a cell
        self.BW = BW  # Bandwidth = 10MHz
        self.NF = NF  # Power of noise figure [dBm]
        self.Ns = Ns  # Number of sample
        self.min_P = min_P  # Minimum transmission power [dBm]
        self.max_P = max_P  # Maximum transmission power [dBm]
        self.num_P = num_P  # Number of action space
