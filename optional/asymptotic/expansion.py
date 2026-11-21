"""Independent implementation of the first-order expansion of Rico and Hermanns.

Section 7 single-borehole benchmark with constant pipe powers. Lengths are in
borehole radii and the temperature scale is q_c/k_ground.
Original method: https://doi.org/10.1016/j.ijthermalsci.2024.109222
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.special import exp1


RADII = np.array([20.0, 25.0]) / 76.0
OFFSETS = np.array([51.0, 46.0]) / 76.0
BEARINGS = np.array([45.0, -45.0])
KAPPA = 1.50 / 3.00
RP = 3.00 * np.array([0.535, 0.544])  # dimensionless k_ground * inner resistance
POWER = np.array([5.0, 1.0])


def multipoles(z, normal, centres, order):
    """Real logarithmic sources and higher poles, each with its circle image.

    Return values and directional derivatives. Leading columns give unit heat
    sources; the remaining columns are real and imaginary multipole amplitudes.
    """
    sigma = (KAPPA - 1) / (KAPPA + 1)
    modes = np.arange(1, order + 1)
    logs, dlogs, homogeneous, dhomogeneous = [], [], [], []
    for c, r in zip(centres, RADII):
        dz = z - c
        image = 1 - c.conjugate() * z
        logs.append(-(np.log(abs(dz)) + sigma * np.log(abs(image))) / (2 * np.pi * KAPPA))
        dlogs.append(-(normal * (1 / dz - sigma * c.conjugate() / image)).real / (2 * np.pi * KAPPA))
        h = (r / dz[:, None]) ** modes
        reflected = (r * z[:, None] / image[:, None]) ** modes
        dh = -modes * h / dz[:, None]
        # Differentiate the regular image analytically, including at z=0.
        dreflected = (
            modes * r * (r * z[:, None] / image[:, None]) ** (modes - 1) / image[:, None] ** 2
        )
        for phase in (1.0, 1j):
            homogeneous.append((phase * h + sigma * np.conj(phase) * reflected).real)
            dhomogeneous.append(
                (normal[:, None] * (phase * dh + sigma * np.conj(phase) * dreflected)).real
            )
    return (
        np.column_stack(logs),
        np.column_stack(dlogs),
        np.column_stack(homogeneous),
        np.column_stack(dhomogeneous),
    )


def endogenous_inner(wall_modes, wall_order):
    """Regular grout particular solution from Appendix A.1-A.3, per unit source.

    For positive harmonic l the interior polynomial coefficient is 2*P_l
    because the conjugate negative harmonic completes the real temperature.
    """
    l = np.arange(1, wall_order + 1)
    g = (wall_modes[l + 1] + wall_modes[l - 1]) / 4
    dg = -((l + 2)[:, None] * wall_modes[l + 1] + (l - 2)[:, None] * wall_modes[l - 1]) / 4
    g[0] = wall_modes[2] / 4
    dg[0] = -1 / (4 * np.pi) - 0.75 * wall_modes[2]  # total power in each unit solve = 1
    poly = np.empty((wall_order + 1, wall_modes.shape[1]), complex)
    poly[0] = wall_modes[1].real / 2
    poly[1:] = 2 * (g + (dg / l[:, None] - KAPPA * g) / (1 + KAPPA))
    return poly


@dataclass
class Inner:
    resistance0: np.ndarray
    resistance1: np.ndarray
    wall_dipole: np.ndarray  # sum of +1 and -1 Fourier coefficients, per source
    gradient_response: np.ndarray  # local pipe response per unit real C in Eq. (24)
    max_robin_residual: float
    max_power_residual: float

    def resistances(self, pe):
        return self.resistance0 + pe * self.resistance1

    def weights(self, pe):
        # Section 6: normalize column sums of the inverse resistance matrix.
        r = self.resistances(pe)
        v = np.linalg.solve(r.T, np.ones(len(r)))
        return v / v.sum()

    def temperatures(self, pe, tau, power=POWER):
        """Two fluid temperatures for positive times; tau=inf is the steady limit."""
        tau = np.asarray(tau, float)
        if not np.isfinite(pe) or pe < 0 or np.any(np.isnan(tau)) or np.any(tau <= 0):
            raise ValueError("Use nonnegative Pe and positive dimensionless times")
        q = np.asarray(power, float)
        total, dipole = q.sum(), self.wall_dipole @ q
        if pe == 0:
            if np.any(np.isinf(tau)):
                raise ValueError("No conductive steady unit-step response")
            common = total * (np.log(4 * tau) - np.euler_gamma) / (4 * np.pi)
            return common[..., None] + self.resistance0 @ q
        # Equations (23) and (24); the exponentially small E1 term vanishes at DC.
        bracket = exp1((pe / 2) ** 2 * tau) + 2 * (np.log(pe / 4) + np.euler_gamma)
        apparent = -(total - pe * np.pi * dipole) * bracket / (4 * np.pi) + pe * dipole / 4
        gradient = -total * bracket / (16 * np.pi)
        return (
            apparent[..., None]
            + pe * gradient[..., None] * self.gradient_response
            + self.resistances(pe) @ q
        )


def inner_problem(rotation=90.0, order=20, wall_order=64, samples=512):
    """Solve the three inner problems with the same projected pipe Robin matrix."""
    if order < 1 or wall_order < 2 or samples < 2 * (max(order, wall_order) + 2):
        raise ValueError("Insufficient boundary resolution")
    centres = OFFSETS * np.exp(1j * np.deg2rad(BEARINGS + rotation))
    unit = np.exp(2j * np.pi * np.arange(samples) / samples)
    z = centres[:, None] + RADII[:, None] * unit
    normals = np.broadcast_to(unit, z.shape).ravel()
    pipe_points = z.ravel()
    row_size = np.repeat(RADII, samples)[:, None]
    robin_scale = np.repeat(KAPPA * RP, samples)[:, None] * row_size
    log, dlog, h, dh = multipoles(pipe_points, normals, centres, order)
    basis_robin = h - robin_scale * dh

    def project(f):
        harmonics = np.fft.rfft(f.reshape(2, samples, -1), axis=1)[:, 1 : order + 1]
        return (
            np.concatenate((harmonics.real, harmonics.imag), axis=1).reshape(4 * order, -1) / samples
        )

    factor = lu_factor(project(basis_robin))
    residuals = []

    def solve_part(part, dpart, powers):
        boundary = part - robin_scale * dpart
        amplitudes = lu_solve(factor, -project(boundary))
        robin = (boundary + basis_robin @ amplitudes).reshape(2, samples, -1)
        fluid = robin.mean(axis=1)
        flux = (
            -2
            * np.pi
            * KAPPA
            * (row_size * (dpart + dh @ amplitudes)).reshape(2, samples, -1).mean(axis=1)
        )
        residuals.append((np.abs(robin - fluid[:, None, :]).max(), np.abs(flux - powers).max()))
        return amplitudes, fluid

    coeff, R0 = solve_part(log, dlog, np.eye(2))
    wall_log, _, wall_h, _ = multipoles(unit, unit, centres, order)
    A = np.fft.rfft(wall_log + wall_h @ coeff, axis=0) / samples
    poly = endogenous_inner(A, wall_order)
    powers = np.arange(wall_order + 1)
    basis = pipe_points[:, None] ** powers
    part = (basis @ poly).real
    dpart = (normals[:, None] * (basis[:, :-1] @ (powers[1:, None] * poly[1:]))).real
    _, R1 = solve_part(part, dpart, np.zeros((2, 2)))

    # Single borehole: aligned outer gradient C is real, both conjugate terms contribute.
    gradient_scale = 4 / (1 + KAPPA)
    _, response = solve_part(
        gradient_scale * pipe_points.real[:, None],
        gradient_scale * normals.real[:, None],
        np.zeros((2, 1)),
    )
    robin_error, power_error = np.max(residuals, axis=0)
    return Inner(R0, R1, 2 * A[1].real, response[:, 0], robin_error, power_error)
