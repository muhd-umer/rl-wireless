"""
Helper functions for the environment
"""
from functools import partial
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
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import erfinv
from scipy.linalg import sqrtm, toeplitz
from numba import jit, njit, prange


def randn2(*args, **kargs):
    args_r = tuple(reversed(args))
    uniform = np.random.rand(*args_r)

    return np.sqrt(2) * erfinv(2 * uniform - 1)


def hermitian(A):
    """
    Hermitian of a matrixs
    """
    return np.swapaxes(np.conj(A), -1, -2)


def mldivide(A, B):
    """
    Divide function
    """
    return hermitian(np.linalg.solve(A, hermitian(B)))


@njit
def corr(
    x: float,
    theta: ArrayLike,
    asd: float,
    antenna_spacing: float,
    col: int,
    real_imag: int,
) -> ArrayLike:
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

    @njit
    def corr_real(x, antenna_spacing, col):
        return np.cos(2 * np.pi * antenna_spacing * col * np.sin(x))

    @njit
    def corr_imag(x, antenna_spacing, col):
        return np.sin(2 * np.pi * antenna_spacing * col * np.sin(x))

    @njit
    def gaussian_pdf(x, mean, dev):
        return np.exp(-((x - mean) ** 2) / (2 * dev**2)) / (np.sqrt(2 * np.pi) * dev)

    if real_imag == 0:
        res = corr_real(x, antenna_spacing, col)
    else:
        res = corr_imag(x, antenna_spacing, col)

    res *= gaussian_pdf(x, theta, asd)

    return res


def dBm_to_W(x):
    """
    Convert dBm to W
    """
    P = 10 ** ((x - 30) / 10)
    return P


def r_scattering(
    M: int,
    theta: ArrayLike,
    asd_deg: float,
    antenna_spacing: float = 0.5,
    dtype=np.complex128,
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
    asd = asd_deg * np.pi / 180
    init_row = np.zeros(
        [
            M,
        ],
        dtype=dtype,
    )

    distance = np.arange(M)
    x1 = np.exp(1j * 2 * np.pi * antenna_spacing * np.sin(theta) * distance)
    x2 = np.exp(
        -(asd**2) / 2 * (2 * np.pi * antenna_spacing * np.cos(theta) * distance) ** 2
    )
    init_row = x1 * x2

    return toeplitz(c=np.conj(init_row))


def complex_normalize(X, axis=-1):
    """
    Normalize complex vector
    """
    mags = np.linalg.norm(np.abs(X), axis=axis, keepdims=True)

    return X / mags


def noise_dbm(BW=10e6, NF=7):
    """
    # Noise figure with 10MHz BW
    """
    return -174 + 10 * np.log10(BW) + NF


def zf_combining(H):
    """
    ZF precoded combination
    """
    H1 = np.asarray(H)
    A = hermitian(H) @ H1 + 1e-12 * np.eye(H1.shape[-1])
    B = H1
    res = mldivide(A, B)

    return res


def channel_setup(
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
        no_BS_per_dim = np.array([np.sqrt(N), np.sqrt(N)])
    inter_bs_distance = side_length / no_BS_per_dim

    # scatter the BSs
    BS_positions = np.stack(
        np.meshgrid(
            np.arange(inter_bs_distance[0] / 2, side_length, inter_bs_distance[0]),
            np.arange(inter_bs_distance[1] / 2, side_length, inter_bs_distance[1]),
            indexing="ij",
        ),
        axis=2,
    ).reshape([-1, 2])

    # now all the other nine alternatives of the BS locations
    wrap_locations = np.stack(
        np.meshgrid(
            np.array([-side_length, 0, side_length]),
            np.array([-side_length, 0, side_length]),
            indexing="ij",
        ),
        axis=2,
    ).reshape([-1, 2])

    # for each BS locations, there are 9 possible
    # alternative locations including the original one.
    # Here uses broadcasting to add (9,2) to a (num_BS, 1, 2)
    # to get a (num_BS, 9, 2)
    BS_positions_wrapped = np.expand_dims(BS_positions, axis=1) + wrap_locations

    UEpositions = np.zeros([K, N, 2])
    perBS = np.zeros(
        [
            N,
        ],
        dtype=np.int32,
    )

    # normalized spatial correlation matrices
    R = np.zeros([M, M, K, N, N, len(asd_degs)], dtype=np.complex128)

    channel_gain = np.zeros([K, N, N])

    for i in range(N):
        # put K UEs in the cell, uniformly. UE's not satisfying
        # the minimum distance are replaced
        res = []
        while perBS[i] < K:
            UEremaining = K - perBS[i]
            pos = np.random.uniform(
                -inter_bs_distance / 2, inter_bs_distance / 2, size=(UEremaining, 2)
            )
            cond = np.linalg.norm(pos, ord=2, axis=1) >= min_UE_BS_dist

            # satisfying minimum distance with respect to BS shape
            pos = pos[cond, :]
            for x in pos:
                res.append(x + BS_positions[i])
            perBS[i] += pos.shape[0]

        UEpositions[:, i, :] = np.array(res)

        # loop through all BS for cross-channels
        for j in range(N):
            # distance between all UEs in cell i to BS j, considering wrap-around.
            dist_ue_i_j = np.linalg.norm(
                np.expand_dims(UEpositions[:, i], axis=1)
                - BS_positions_wrapped[j, :, :],
                axis=2,
            )
            dist_bs_j = np.min(dist_ue_i_j, axis=1)
            which_pos = np.argmin(dist_ue_i_j, axis=1)

            # average channel gain with large-scale fading mdoel in (2.3), neglecting shadow fading
            channel_gain[:, i, j] = constant_term - alpha * 10 * np.log10(dist_bs_j)

            # generate spatial correlation matrices for channels with local scattering model
            for k in range(K):
                vec_ue_bs = UEpositions[k, i] - BS_positions_wrapped[j, which_pos[k]]
                angle_BS_j = np.arctan2(vec_ue_bs[1], vec_ue_bs[0])
                for spr, asd_deg in enumerate(asd_degs):
                    R[:, :, k, i, j, spr] = r_scattering(
                        M, angle_BS_j, asd_deg, antenna_spacing
                    )

        # all UEs in cell i to generate shadow fading realizations
        for k in range(K):
            # see if another BS has a larger avg. channel gain to the UE than BS i
            while True:
                # generate new shadow fading realizations until
                # all UE's in cell i has its largest avg. channel gain from BS i
                shadowing = np.array(sigma_sf * randn2(N))
                channel_gain_shadowing = channel_gain[k, i] + shadowing
                if channel_gain_shadowing[i] >= np.max(channel_gain_shadowing):
                    break
            channel_gain[k, i, :] = channel_gain_shadowing

    return R, channel_gain


def get_channnel_realization(M, K, N, Ns):
    """
    Generate uncorrelated Rayleigh fading channel realizations with unit variance

    Args:
        M: number of antennas at BS
        K: number of UEs
        N: number of BSs
        Ns: number of channel realizations

    Returns:
        H: channel realizations
    """
    randn2 = np.random.randn
    H = randn2(N, N, K, M, Ns) + 1j * randn2(N, N, K, M, Ns)

    return np.sqrt(0.5) * H


def get_local_scatter(
    N: int,
    M: int,
    K: int,
    Ns: int,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Generate local scattering channel matrix

    Args:
        key: random key
        N: number of antennas at BS
        M: number of UEs
        K: number of BSs
        Ns: number of channel realizations

    Returns:
        res: channel matrix
        H_gain: channel matrix with gain
    """

    if N > 1 and N < 4:
        no_BS_per_dim = np.array([1, N])
    else:
        no_BS_per_dim = None
    R, gain_db = channel_setup(
        N,
        K,
        M,
        no_BS_per_dim=no_BS_per_dim,
        asd_degs=[
            30,
        ],
    )
    gain_db -= noise_dbm()

    R_gain = R[:, :, :, :, :, 0] * np.power(10, gain_db / 10.0)
    R_gain = np.ascontiguousarray(np.transpose(R_gain[:, :, :, :, :], (4, 3, 2, 1, 0)))

    H = get_channnel_realization(M, K, N, Ns)
    H_gain = np.zeros_like(H)
    for _idx in np.ndindex(*H.shape[0:3]):
        H_gain[_idx] = sqrtm(R_gain[_idx]) @ H[_idx]

    res = np.ascontiguousarray(np.transpose(H_gain, (4, 0, 1, 2, 3)))

    return res, H_gain


def get_precoding(H):
    """
    ZF precoding method

    Args:
        H: channel matrix
        local_cell_info: whether to use local cell information

    Returns:
        W: precoding matrix
    """
    res = []
    no_cells = H.shape[1]
    for j in range(no_cells):
        res.append(zf_combining(H[:, j, j]))

    return np.stack(res, axis=1)


def antenna_selection(H, ant_sel=False):
    """
    Antenna selection and power allocation by using ZF precoding

    Args:
        H: channel matrix
        ant_sel: antenna selection

    Returns:
        W: power allocation
    """
    Ns, N, N, K, M = H.shape

    if ant_sel:
        antenna_sel = np.zeros((K, N, Ns, N, M), dtype=np.bool_)

        # strongest K_0 antennas
        K_0 = int(M * 0.8)
        for r in range(Ns):
            for n in range(N):
                channel_power_ant = (np.abs(H[r, n, n]) ** 2).sum(axis=-2)
                top_k = np.argsort(channel_power_ant)[0:K_0]
                antenna_sel[:, :, r, n, top_k] = True
        # or randomly
        antenna_sel = antenna_sel * 1.0
        H_n = np.transpose(
            (np.transpose(H, (2, 0, 4, 1, 3)) * antenna_sel), (2, 3, 0, 1, 4)
        )

    else:
        antenna_sel = np.ones((Ns, N, M), dtype=np.bool_)
        H_n = H

    W = get_precoding(H_n)

    return W
