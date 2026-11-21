"""Fixed Euler-accelerated Bromwich quadrature; no trainable inversion weights."""

import math
import numpy as np


def nodes(t, A=18.0, n=20, m=20):
    t = np.asarray(t, float)
    assert np.all(t > 0)
    k = np.arange(n + m + 1)
    s = (A + 2j * np.pi * k[None, :]) / (2 * t.reshape(-1, 1))
    # Euler mean of partial alternating sums; polynomial quadrature weights fixed a priori.
    euler = np.array([math.comb(m, j) / 2**m for j in range(m + 1)])
    weight = np.array([sum(euler[max(0, k_ - n) :]) * (-1) ** k_ for k_ in k])
    weight[0] *= 0.5
    return s, weight * np.exp(A / 2) / t.reshape(-1, 1)
