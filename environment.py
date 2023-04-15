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

        # Channel parameters and matrices
        # self.precoding_matrix = self.antenna_selection()
        # self.H, self.H_gain = self.get_channel_mats(no_realization=self.Ns)

        self.power_alloc = self.get_power_alloc()  # Power allocation (action value)
        self.num_actions = len(self.power_alloc)  # Number of actions
        self.action_space = Discrete(self.num_actions)  # Action space
        mem_limits = jnp.finfo(jnp.float32).max  # Maximum value of float32

        # Observation space: contains transmission power, SINR, and sum rate
        self.observation_space = Box(
            low=-mem_limits, high=mem_limits
        )  # Observation space

        # Initialize the state
        self.state = None
        self.downlink_rate = None
        self.rate_list = []

    """
    Downlink Scenario
    """

    def get_power_alloc(self):
        """
        Get power allocation
        """
        power_alloc = jnp.hstack(
            [
                jnp.zeros((1), dtype=self.dtype),
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

    @jit
    def mldivide(self, A, B):
        """
        Divide function
        """
        return self.hermitian(jnp.linalg.solve(A, self.hermitian(B)))

    def randn_sqr(self, *args, **kargs):
        """
        Random number generator
        """
        args_r = tuple(reversed(args))
        key = jax.random.PRNGKey(0)
        uniform = jax.random.uniform(key, shape=args_r, dtype=self.dtype)

        return jnp.sqrt(2) * erfinv(2 * uniform - 1)

    @partial(jit, static_argnames=["dist", "real_imag"])
    def corr(x, theta, asd, antenna_spacing, dist, col, real_imag):
        """
        Correlation function
        """

        def corr_real(x, antenna_spacing, col):
            return jnp.cos(2 * jnp.pi * antenna_spacing * col * jnp.sin(x))

        def corr_imag(x, antenna_spacing, col):
            return jnp.sin(2 * jnp.pi * antenna_spacing * col * jnp.sin(x))

        def gaussian_pdf(x, mean, dev):
            return jnp.exp(-((x - mean) ** 2) / (2 * dev**2)) / (
                jnp.sqrt(2 * jnp.pi) * dev
            )

        if real_imag == 0:
            res = corr_real(x, antenna_spacing, col)
        else:
            res = corr_imag(x, antenna_spacing, col)
        if dist == "gaussian":
            res *= gaussian_pdf(x, theta, asd)

        return res

    @jit
    def dBm_to_W(self, x):
        """
        Convert dBm to W
        """
        P = 10 ** ((x - 30) / 10)
        return P

    def r_scattering(
        self,
        M,
        theta,
        asd_deg,
        antenna_spacing=0.5,
        dist="gaussian",
        accuracy=1,
        dtype=jnp.complex128,
    ):
        """
        Local scattering model of a single cell
        Input:
            M: number of antennas
            theta: angle of arrival
            asd_deg: angular standard deviation in degrees
            antenna_spacing: antenna spacing in wavelengths
            dist: distribution of AoA
            accuracy: accuracy of the model
            dtype: data type
        Output:
            R: correlation matrix
        """
        asd = asd_deg * jnp.pi / 180
        init_row = jnp.zeros(
            [
                M,
            ],
            dtype=dtype,
        )

        if accuracy == 1:
            lb = None
            ub = None

            dist = dist.lower()
            if dist == "gaussian":
                lb = theta - 20 * asd
                ub = theta + 20 * asd

            else:
                raise NotImplementedError

            for col in range(0, M):
                c_real: float = quad(
                    func=self.corr,
                    a=lb,
                    b=ub,
                    args=(theta, asd, antenna_spacing, dist, col, 0),
                )[0]
                c_imag: float = quad(
                    func=self.corr,
                    a=lb,
                    b=ub,
                    args=(theta, asd, antenna_spacing, dist, col, 1),
                )[0]

                init_row[col] = complex(c_real, c_imag)

        elif accuracy == 2:
            distance = jnp.arange(M)
            x1 = jnp.exp(1j * 2 * jnp.pi * antenna_spacing * jnp.sin(theta) * distance)
            x2 = jnp.exp(
                -(asd**2)
                / 2
                * (2 * jnp.pi * antenna_spacing * jnp.cos(theta) * distance) ** 2
            )
            init_row = x1 * x2

        return toeplitz(c=init_row.conjugate())

    @jit
    def complex_normalize(self, X, axis=-1):
        """
        Normalize complex vector
        """
        mags = np.linalg.norm(np.abs(X), axis=axis, keepdims=True)

        return X / mags

    def noise_dbm(
        self,
    ):
        """
        # Noise figure with 20MHz BW
        """
        return -174 + 10 * np.log10(self.BW) + self.NF

    @jit
    def zf_combining(self, H):
        """
        ZF precoded combination
        """
        H1 = H
        A = self.hermitian(H1) @ H1 + 1e-12 * np.eye(H1.shape[-1])
        B = H1
        res = self.mldivide(A, B)

        return res
    