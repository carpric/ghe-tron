"""Two prescribed pipe powers and the radially refined low-frequency response."""

import numpy as np
from model.boundary import Exterior
from model.response import Circuit, MultiPipeResponse


def exchanger(pe, rotation, table=None):
    angles = np.deg2rad(np.array([45.0, -45.0]) + rotation)
    xy = np.array([51.0, 46.0])[:, None] * np.c_[np.cos(angles), np.sin(angles)] / 76.0
    r = np.array([20.0, 25.0]) / 76.0
    k_grout, k_ground = 1.50, 3.00
    rp = np.array([0.535, 0.544])
    beta = k_grout * rp * r
    # In the authors' units Delta T = q_c/k_ground and the total is 6 q_c.
    coupling = 2 * np.pi * k_grout / (6 * k_ground)
    op = MultiPipeResponse(
        xy, r, beta, coupling, 0.5, 2.24, pe, 0.0, Circuit.prescribed_heat_fractions([5 / 6, 1 / 6]), table
    )
    np.testing.assert_allclose(op.right, k_ground * rp * np.array([5, 1]) / (2 * np.pi), atol=1e-13)
    return op


def dc(op, s):
    inc, _ = op.frequency(complex(s), False)
    if not np.isfinite(inc).all() or np.max(abs(inc.imag)) > 3e-8:
        raise ArithmeticError("Invalid real-frequency response")
    return op.right + inc.real


INTERVALS = (28, 56, 112, 160, 224, 320)
TOL = 1e-6
S = 1e-10


def converged_temperatures(pe):
    op = exchanger(pe, 90)
    values = []
    for intervals in INTERVALS:
        op.exterior = Exterior(pe, op.m, intervals, 14)
        values.append(dc(op, S))
    values = np.asarray(values)
    for idx, intervals in enumerate(INTERVALS[:-1]):
        finer = values[idx + 1 :]
        difference = np.max(abs(values[idx] - finer) / np.maximum(1.0, abs(finer)), axis=0)
        if np.max(difference) <= TOL:
            op.exterior = Exterior(pe, op.m, intervals, 14)
            lower_s = dc(op, S / 10)
            limit_change = abs(values[idx] - lower_s) / np.maximum(1.0, abs(lower_s))
            if np.max(limit_change) > TOL:
                raise ArithmeticError(f"Small-frequency limit unresolved at Pe={pe}")
            return idx, values, difference, limit_change
    raise ArithmeticError(f"Exterior grid unresolved at Pe={pe}")
