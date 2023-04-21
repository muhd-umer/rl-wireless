"""
Helper functions for the environment
"""

from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from jaxtyping import Array, Float, ArrayLike
import numpy as np
from scipy.integrate import quad
from scipy.special import erfinv
from jax.scipy.linalg import toeplitz


def randn2(self, *args, **kargs):
    args_r = tuple(reversed(args))
    uniform = np.random.rand(*args_r)

    return np.sqrt(2) * erfinv(2 * uniform - 1)


@jit
def hermitian(A: Array) -> Array:
    """
    Hermitian of a matrixs
    """
    return jnp.transpose(jnp.conj(A))


@jit
def mldivide(A: Array, B: Array) -> Array:
    """
    Divide function
    """
    return hermitian(jnp.linalg.solve(A, hermitian(B)))


@partial(jit, static_argnames=["real_imag"])
def corr(
    x: float,
    theta: ArrayLike,
    asd: float,
    antenna_spacing: float,
    col: int,
    real_imag: int,
) -> Array:
    """
    Correlation function
    Args:
        x: angle of arrival
        theta: angle of arrival
        asd: angular standard deviation
        antenna_spacing: antenna spacing
        col: column index
        real_imag: real or imaginary part
    Returns:
        res: correlation value
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

    res *= gaussian_pdf(x, theta, asd)

    return res


@jit
def dBm_to_W(x):
    """
    Convert dBm to W
    """
    P = 10 ** ((x - 30) / 10)
    return P


@partial(jit, static_argnames=["M", "dtype"])
def r_scattering(
    M: int,
    theta: ArrayLike,
    asd_deg: float,
    antenna_spacing: float = 0.5,
    dtype=jnp.complex64,
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

    distance = jnp.arange(M)
    x1 = jnp.exp(1j * 2 * jnp.pi * antenna_spacing * jnp.sin(theta) * distance)
    x2 = jnp.exp(
        -(asd**2)
        / 2
        * (2 * jnp.pi * antenna_spacing * jnp.cos(theta) * distance) ** 2
    )
    init_row = x1 * x2

    return toeplitz(c=jnp.conj(init_row))


@jit
def complex_normalize(X, axis=-1):
    """
    Normalize complex vector
    """
    mags = jnp.linalg.norm(jnp.abs(X), axis=axis, keepdims=True)

    return X / mags


def noise_dbm(BW, NF):
    """
    # Noise figure with 20MHz BW
    """
    return -174 + 10 * jnp.log10(BW) + NF


@jit
def zf_combining(H):
    """
    ZF precoded combination
    """
    H1 = H
    A = hermitian(H) @ H1 + 1e-12 * jnp.eye(H.shape[-1])
    B = H
    res = mldivide(A, B)

    return res


def channel_setup(
    key,
    N: int,
    K: int,
    M: int,
    asd_degs: List,
    no_BS_per_dim: Any = None,
):
    """
    Channel statistics between UE's at random locations and the BS.

    Args:
        N: number of UEs
        K: number of BSs
        M: number of antennas
        asd_degs: angular standard deviation in degrees
        no_BS_per_dim: number of BSs per dimension
    Returns:
        R: correlation matrix
        channel_gain: channel gain
    """
    side_length = 500  # square side, in meters
    sigma_sf = 10  # standard deviation of shadow fading
    min_UE_BS_dist = 25  # minimum distance between BS and UEs
    max_UE_BS_dist = 300  # maximum distance between BS and UEs

    alpha = 3.76  # pathloss exp
    constant_term = (
        -35.3
    )  # avg. channel gain: At exponent set to 3.76, at 1km it's -148.1 dB

    # antenna spacing # of wavelengths
    antenna_spacing = 0.5
    if no_BS_per_dim is None:
        no_BS_per_dim = jnp.array([jnp.sqrt(N), jnp.sqrt(N)])
    inter_bs_distance = side_length / no_BS_per_dim

    # scatter the BSs
    BS_positions = jnp.stack(
        jnp.meshgrid(
            jnp.arange(inter_bs_distance[0] / 2, side_length, inter_bs_distance[0]),
            jnp.arange(inter_bs_distance[1] / 2, side_length, inter_bs_distance[1]),
            indexing="ij",
        ),
        axis=2,
    ).reshape([-1, 2])

    # now all the other nine alternatives of the BS locations
    wrap_locations = jnp.stack(
        jnp.meshgrid(
            jnp.array([-side_length, 0, side_length]),
            jnp.array([-side_length, 0, side_length]),
            indexing="ij",
        ),
        axis=2,
    ).reshape([-1, 2])

    # for each BS locations, there are 9 possible
    # alternative locations including the original one.
    # Here uses broadcasting to add (9,2) to a (num_BS, 1, 2)
    # to get a (num_BS, 9, 2)
    BS_positions_wrapped = jnp.expand_dims(BS_positions, axis=1) + wrap_locations

    UEpositions = jnp.zeros([K, N, 2])
    perBS = np.zeros(
        [
            N,
        ],
        dtype=np.int32,
    )

    # normalized spatial correlation matrices
    R = jnp.zeros([M, M, K, N, N, len(asd_degs)], dtype=jnp.complex64)

    channel_gain = jnp.zeros([K, N, N])

    for i in range(N):
        # put K UEs in the cell, uniformly. UE's not satisfying
        # the minimum distance are replaced
        res = []
        while perBS[i] < K:
            key, subkey = jax.random.split(key)
            UEremaining = K - perBS[i]
            pos = jax.random.uniform(
                key=subkey,
                minval=-inter_bs_distance / 2,
                maxval=inter_bs_distance / 2,
                shape=(UEremaining, 2),
            )
            cond = jnp.linalg.norm(pos, ord=2, axis=1) >= min_UE_BS_dist

            # satisfying minimum distance with respect to BS shape
            pos = pos[cond, :]
            for x in pos:
                res.append(x + BS_positions[i])
            perBS[i] += pos.shape[0]

        UEpositions = UEpositions.at[:, i, :].set(jnp.array(res))

        # loop through all BS for cross-channels
        for j in range(N):
            # distance between all UEs in cell i to BS j, considering wrap-around.
            dist_ue_i_j = jnp.linalg.norm(
                jnp.expand_dims(UEpositions[:, i], axis=1)
                - BS_positions_wrapped[j, :, :],
                axis=2,
            )
            dist_bs_j = jnp.min(dist_ue_i_j, axis=1)
            which_pos = jnp.argmin(dist_ue_i_j, axis=1)

            # average channel gain with large-scale fading mdoel in (2.3), neglecting shadow fading
            channel_gain = channel_gain.at[:, i, j].set(
                constant_term - alpha * 10 * jnp.log10(dist_bs_j)
            )

            # generate spatial correlation matrices for channels with local scattering model
            for k in range(K):
                vec_ue_bs = UEpositions[k, i] - BS_positions_wrapped[j, which_pos[k]]
                angle_BS_j = jnp.arctan2(vec_ue_bs[1], vec_ue_bs[0])
                for spr, asd_deg in enumerate(asd_degs):
                    R = R.at[:, :, k, i, j, spr].set(
                        r_scattering(M, angle_BS_j, asd_deg, antenna_spacing)
                    )

        # all UEs in cell i to generate shadow fading realizations
        for k in range(K):
            # see if another BS has a larger avg. channel gain to the UE than BS i
            while True:
                # generate new shadow fading realizations until
                # all UE's in cell i has its largest avg. channel gain from BS i
                shadowing = jnp.array(sigma_sf * randn2(N))
                channel_gain_shadowing = channel_gain[k, i] + shadowing
                if channel_gain_shadowing[i] >= jnp.max(channel_gain_shadowing):
                    break
            channel_gain = channel_gain.at[k, i, :].set(channel_gain_shadowing)

    return R, channel_gain
