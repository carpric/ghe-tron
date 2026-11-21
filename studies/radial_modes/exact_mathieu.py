"""Experiment: closed-form radial modes of the transient exterior problem.

Radial equation (log-polar, zeta = log r):
    R'' = [a + (s + k^2) e^{2 zeta} + k^2 e^{-2 zeta}] R
For a nonvanishing unshifted product, R(r) = sum_j c_j K_{nu+j}(p r) I_j(k / r),  p = sqrt(s + k^2),
where nu, c_j are the Floquet exponent and coefficients of the ordinary
Mathieu equation w'' + (a - 2 q cos 2z) w = 0 with q = k p and the SAME a
(the angular eigenvalue for q_ang = -k^2), i.e.
    ((nu + 2j)^2 - a) c_j + q (c_{j-1} + c_{j+1}) = 0.
"""
import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

from model.boundary import compact_grid  # the repository's own collocation grid

mp.mp.dps = 40


def angular_eigs(k, nmax=40):
    n = np.arange(-nmax, nmax + 1)
    H = np.diag(n.astype(float) ** 2) - k**2 * (np.abs(n[:, None] - n) == 2)
    return eigh(H, eigvals_only=True)


def floquet_nu(a, q):
    """cos(pi nu) = w_I(pi; a, q) for w'' + (a - 2 q cos 2z) w = 0 (DLMF 28.2.16)."""
    a, q = mp.mpc(a), mp.mpc(q)
    f = lambda z, y: [y[1], -(a - 2 * q * mp.cos(2 * z)) * y[0]]
    sol = mp.odefun(f, 0, [mp.mpf(1), mp.mpf(0)])
    w = sol(mp.pi)[0]
    return mp.acos(w) / mp.pi


def floquet_coeffs(a, q, nu, J):
    j = np.arange(-J, J + 1)
    M = mp.matrix(len(j), len(j))
    for i, jj in enumerate(j):
        M[i, i] = (nu + 2 * jj) ** 2 - a
        if i > 0:
            M[i, i - 1] = q
        if i < len(j) - 1:
            M[i, i + 1] = q
    # null vector via inverse iteration on the nearly singular matrix
    v = mp.matrix([1] * len(j))
    shift = mp.mpf("1e-30")
    for _ in range(6):
        v = mp.lu_solve(M + shift * mp.eye(len(j)), v)
        v = v / mp.norm(v)
    resid = mp.norm(M * v) / mp.norm(v)
    return j, v, resid


def exact_Z(a, k, s, J=30):
    """Z = -r R'(r)/R(r) at r = 1 from the Bessel-product series."""
    p = mp.sqrt(mp.mpc(s) + k**2)
    q = k * p
    nu = floquet_nu(a, q)
    j, c, resid = floquet_coeffs(a, q, nu, J)
    R = dR = mp.mpc(0)
    for cj, jj in zip(c, j):
        K = mp.besselk(nu + jj, p)
        dK = -(mp.besselk(nu + jj - 1, p) + mp.besselk(nu + jj + 1, p)) / 2
        I = mp.besseli(abs(int(jj)), k)
        dI = (mp.besseli(abs(int(jj)) - 1, k) + mp.besseli(abs(int(jj)) + 1, k)) / 2
        R += cj * K * I
        dR += cj * (p * dK * I - k * K * dI)  # r dR/dr at r = 1
    return complex(-dR / R), complex(nu), float(resid)


def series_residual(a, k, s, r, J=30):
    """ODE residual of the series at radius r (checks the closed form itself)."""
    p = mp.sqrt(mp.mpc(s) + k**2)
    q = k * p
    nu = floquet_nu(a, q)
    j, c, _ = floquet_coeffs(a, q, nu, J)

    def R(z):  # z = log r
        rr = mp.e**z
        return mp.fsum(cj * mp.besselk(nu + jj, p * rr) * mp.besseli(abs(int(jj)), k / rr) for cj, jj in zip(c, j))

    z0 = mp.log(r)
    lhs = mp.diff(R, z0, 2)
    rhs = (a + (mp.mpc(s) + k**2) * mp.e ** (2 * z0) + k**2 * mp.e ** (-2 * z0)) * R(z0)
    return complex(abs(lhs - rhs) / abs(rhs))


def collocation_Z(a, k, s, radial):
    """The repository's compactified collocation, Eqs. (15)-(16) and (18)."""
    x, dx = compact_grid(radial)
    p = np.sqrt(complex(s) + k**2)
    base = x[:, None] ** 2 * (dx @ dx) + 2 * x[:, None] * dx + np.diag(0.25 - k**2 * x * x)
    op = base + 2 * p * dx
    A = op[:-1, :-1] - a * np.eye(len(x) - 1)
    u = np.append(np.linalg.solve(A, -op[:-1, -1]), 1.0)
    return p + 0.5 + u @ dx[-1]


def shooting_Z(a, k, s, L=60.0):
    """Independent check: integrate R from far field inward in zeta."""
    p = np.sqrt(complex(s) + k**2)
    rmax = 1 + L / p.real
    z1 = np.log(rmax)
    V = lambda z: a + (s + k**2) * np.exp(2 * z) + k**2 * np.exp(-2 * z)

    def f(z, y):
        R = y[0] + 1j * y[1]
        dR = y[2] + 1j * y[3]
        d2 = V(z) * R
        return [dR.real, dR.imag, d2.real, d2.imag]

    # WKB start for the decaying branch: R'/R = -sqrt(V)
    g = -np.sqrt(V(z1))
    sol = solve_ivp(f, [z1, 0.0], [1, 0, g.real, g.imag], rtol=1e-12, atol=1e-14, method="DOP853")
    if not sol.success:
        raise RuntimeError(sol.message)
    R = sol.y[0, -1] + 1j * sol.y[1, -1]
    dR = sol.y[2, -1] + 1j * sol.y[3, -1]
    return -dR / R
