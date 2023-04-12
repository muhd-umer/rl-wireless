"""
Implementation of the environment for the RL algorithm.
"""
import numpy as np
from scipy.special import j0
dtype = np.float32

class Environment:
    """
    Environment for the RL algorithm
    """

    def __init__(
        self, fd, Ts, L, C, n_x, n_y, min_dis, max_dis, max_M, max_P, P_n
    ) -> None:
        self.fd = fd  # doppler frequency
        self.Ts = Ts  # adjacent time slot
        self.n_x = n_x  # BS number in x direction
        self.n_y = n_y  # BS number in y direction
        self.L = L  # channel length
        self.C = C  # channel capacity
        self.max_M = max_M  # max user number per BS
        self.max_P = max_P  # dBm max transmit power
        self.P_n = P_n  # dBm noise power

        self.min_dis = min_dis  # min distance between BS and user
        self.max_dis = max_dis  # max distance between BS and user
        self.c = 3 * self.L * (self.L + 1) + 1  # adjacent base station
        self.N = self.n_x * self.n_y  # BS number
        self.M = self.max_M * self.N  # maximum user number
        self.K = self.max_M * self.c  # user number in adjacent BS
        self.sigma_sqr = 1e-3 * pow(10.0, self.P_n / 10.0)  # noise power
        self.max_P_linear = 1e-3 * pow(10.0, self.max_P / 10.0)  # max transmit power
        self.rho = j0(2 * np.pi * self.fd * self.Ts)  # doppler factor
        self.p_array, self.p_list = self.generate_environment()  # allocation function

    def get_power_set(self, min_p):
        power_set = np.hstack(
            [
                np.zeros((1), dtype=dtype),
                1e-3
                * pow(10.0, np.linspace(min_p, self.max_p, self.power_num - 1) / 10.0),
            ]
        )
        return power_set

    def set_Ns(self, Ns):
        self.Ns = Ns  # number of time slots

    # get channel gain matrix
    def set_g(self):
        # h is a complex Gaussian random variable with Rayleigh distributed magnitude
        h = np.zeros((self.M, self.K, self.Ns), dtype=dtype)

        h[:, :, 0] = np.kron(
            np.sqrt(
                0.5
                * (
                    np.random.randn(self.M, self.c) ** 2
                    + np.random.randn(self.M, self.c) ** 2
                )
            ),
            np.ones((1, self.max_M), dtype=np.int32),
        )

        for i in range(1, self.Ns):
            h[:, :, i] = h[:, :, i - 1] * self.rho + np.sqrt(
                (1.0 - self.rho**2)
                * 0.5
                * (
                    np.random.randn(self.M, self.K) ** 2
                    + np.random.randn(self.M, self.K) ** 2
                )
            )

        path_loss = self.get_path_loss()
        g = np.square(h) * path_loss

        return g

    # define the environment
    def generate_environment(self):
        path_matrix = self.M * np.ones(
            (self.n_y + 2 * self.L, self.n_x + 2 * self.L, self.max_M), dtype=np.int32
        )
        for i in range(self.L, self.n_y + self.L):
            for j in range(self.L, self.n_x + self.L):
                for l in range(self.max_M):
                    path_matrix[i, j, l] = (
                        (i - self.L) * self.n_x + (j - self.L)
                    ) * self.max_M + l
        p_array = np.zeros((self.M, self.K), dtype=np.int32)
        for n in range(self.N):
            i = n // self.n_x
            j = n % self.n_x
            Jx = np.zeros((0), dtype=np.int32)
            Jy = np.zeros((0), dtype=np.int32)
            for u in range(i - self.L, i + self.L + 1):
                v = 2 * self.L + 1 - np.abs(u - i)
                jx = (
                    j
                    - (v - i % 2) // 2
                    + np.linspace(0, v - 1, num=v, dtype=np.int32)
                    + self.L
                )
                jy = np.ones((v), dtype=np.int32) * u + self.L
                Jx = np.hstack((Jx, jx))
                Jy = np.hstack((Jy, jy))
            for l in range(self.max_M):
                for k in range(self.c):
                    for u in range(self.max_M):
                        p_array[n * self.max_M + l, k * self.max_M + u] = path_matrix[
                            Jy[k], Jx[k], u
                        ]
        p_main = p_array[
            :, (self.c - 1) // 2 * self.max_M : (self.c + 1) // 2 * self.max_M
        ]
        for n in range(self.N):
            for l in range(self.max_M):
                temp = p_main[n * self.max_M + l, l]
                p_main[n * self.max_M + l, l] = p_main[n * self.max_M + l, 0]
                p_main[n * self.max_M + l, 0] = temp
        p_inter = np.hstack(
            [
                p_array[:, : (self.c - 1) // 2 * self.max_M],
                p_array[:, (self.c + 1) // 2 * self.max_M :],
            ]
        )
        p_array = np.hstack([p_main, p_inter])
        p_list = list()
        for m in range(self.M):
            p_list_temp = list()
            for k in range(self.K):
                p_list_temp.append([p_array[m, k]])
            p_list.append(p_list_temp)

        return p_array, p_list

    # large-scale fading component, taking both geometric attenuation and shadow fading into account
    def get_path_loss(self):
        P_tx = np.zeros((self.n_y, self.n_x))
        P_ty = np.zeros((self.n_y, self.n_x))
        P_rx = np.zeros((self.n_y, self.n_x, self.max_M))
        P_ry = np.zeros((self.n_y, self.n_x, self.max_M))

        dis_rx = np.random.uniform(
            self.min_dis, self.max_dis, size=(self.n_y, self.n_x, self.max_M)
        )
        phi_rx = np.random.uniform(-np.pi, np.pi, size=(self.n_y, self.n_x, self.max_M))

        # geometric attenuation
        for i in range(self.n_y):
            for j in range(self.n_x):
                P_tx[i, j] = 2 * self.max_dis * j + (i % 2) * self.max_dis
                P_ty[i, j] = np.sqrt(3.0) * self.max_dis * i
                for k in range(self.max_M):
                    P_rx[i, j, k] = P_tx[i, j] + dis_rx[i, j, k] * np.cos(
                        phi_rx[i, j, k]
                    )
                    P_ry[i, j, k] = P_ty[i, j] + dis_rx[i, j, k] * np.sin(
                        phi_rx[i, j, k]
                    )
        dis = 1e10 * np.ones((self.p_array.shape[0], self.K), dtype=dtype)
        lognormal = np.random.lognormal(size=(self.p_array.shape[0], self.K), sigma=8)

        # shadow fading
        for k in range(self.p_array.shape[0]):
            for i in range(self.c):
                for j in range(self.max_M):
                    if self.p_array[k, i * self.max_M + j] < self.M:
                        bs = self.p_array[k, i * self.max_M + j] // self.max_M
                        dx2 = np.square(
                            (
                                P_rx[k // self.max_M // self.n_x][
                                    k // self.max_M % self.n_x
                                ][k % self.max_M]
                                - P_tx[bs // self.n_x][bs % self.n_x]
                            )
                        )
                        dy2 = np.square(
                            (
                                P_ry[k // self.max_M // self.n_x][
                                    k // self.max_M % self.n_x
                                ][k % self.max_M]
                                - P_ty[bs // self.n_x][bs % self.n_x]
                            )
                        )
                        distance = np.sqrt(dx2 + dy2)
                        dis[k, i * self.max_M + j] = distance

        path_loss = lognormal * pow(10.0, -(120.9 + 37.6 * np.log10(dis)) / 10.0)

        return path_loss

    # calculate rate matrix
    def calculate_rate(self, P):
        max_C = 1000.0

        g_inter = self.g[:, :, self.count]

        p_extend = np.concatenate([P, np.zeros((1), dtype=dtype)], axis=0)
        p_matrix = p_extend[self.p_array]
        path_main = g_inter[:, 0] * p_matrix[:, 0]
        path_inter = np.sum(g_inter[:, 1:] * p_matrix[:, 1:], axis=1)
        sinr = np.minimum(path_main / (path_inter + self.sigma2), max_C)
        rate = self.W * np.log2(1.0 + sinr)

        sinr_norm_inv = g_inter[:, 1:] / np.tile(g_inter[:, 0:1], [1, self.K - 1])
        sinr_norm_inv = np.log2(1.0 + sinr_norm_inv)  # log representation
        rate_extend = np.concatenate([rate, np.zeros((1), dtype=dtype)], axis=0)
        rate_matrix = rate_extend[self.p_array]

        sum_rate = np.mean(rate)
        reward_rate = rate + np.sum(rate_matrix, axis=1)

        return p_matrix, rate_matrix, reward_rate, sum_rate

    def generate_next_state(self, g_inter, p_matrix, rate_matrix):
        """
        Generate state for actor ranking
        """
        sinr_norm_inv = g_inter[:, 1:] / np.tile(g_inter[:, 0:1], [1, self.K - 1])
        sinr_norm_inv = np.log2(1.0 + sinr_norm_inv)  # log representation
        indices1 = np.tile(
            np.expand_dims(
                np.linspace(
                    0, p_matrix.shape[0] - 1, num=p_matrix.shape[0], dtype=np.int32
                ),
                axis=1,
            ),
            [1, self.C],
        )
        indices2 = np.argsort(sinr_norm_inv, axis=1)[:, -self.C :]
        sinr_norm_inv = sinr_norm_inv[indices1, indices2]
        p_last = np.hstack([p_matrix[:, 0:1], p_matrix[indices1, indices2 + 1]])
        rate_last = np.hstack(
            [rate_matrix[:, 0:1], rate_matrix[indices1, indices2 + 1]]
        )

        s_actor_next = np.hstack([sinr_norm_inv, p_last, rate_last])
        s_critic_next = g_inter

        return s_actor_next, s_critic_next

    def reset(self):
        self.count = 0
        self.g = self.set_g()
        P = np.zeros([self.M], dtype=dtype)

        p_matrix, rate_matrix, _, _ = self.calculate_rate(P)
        g_inter = self.g[:, :, self.count]
        s_actor, s_critic = self.generate_next_state(g_inter, p_matrix, rate_matrix)

        return s_actor, s_critic

    def step(self, P):
        p_matrix, rate_matrix, reward_rate, sum_rate = self.calculate_rate(P)
        self.count = self.count + 1
        H2_next = self.g[:, :, self.count]
        s_actor_next, s_critic_next = self.generate_next_state(
            H2_next, p_matrix, rate_matrix
        )

        return s_actor_next, s_critic_next, reward_rate, sum_rate

    def calculate_sumrate(self, P):
        max_C = 1000.0

        g_inter = self.g[:, :, self.count]
        p_extend = np.concatenate([P, np.zeros((1), dtype=dtype)], axis=0)
        p_matrix = p_extend[self.p_array]
        path_main = g_inter[:, 0] * p_matrix[:, 0]
        path_inter = np.sum(g_inter[:, 1:] * p_matrix[:, 1:], axis=1)
        sinr = np.minimum(path_main / (path_inter + self.sigma2), max_C)  # capped sinr
        rate = self.W * np.log2(1.0 + sinr)
        sum_rate = np.mean(rate)

        return sum_rate

    def step__(self, P):
        reward_rate = list()
        for p in P:
            reward_rate.append(self.calculate_sumrate(p))
        self.count = self.count + 1
        H2_next = self.g[:, :, self.count]

        return H2_next, reward_rate

    def reset__(self):
        self.count = 0
        self.g = self.set_g()
        g_inter = self.g[:, :, self.count]

        return g_inter
