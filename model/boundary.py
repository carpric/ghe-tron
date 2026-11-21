"""Transient multipoles coupled to the decaying separated ground resolvent."""

from time import perf_counter

import numpy as np
from scipy.fft import dct
from scipy.linalg import eigh, schur, solve, solve_banded
from scipy.special import iv, ive, kve


def compact_grid(n):
    x = (1 - np.cos(np.pi * np.arange(n + 1) / n)) / 2
    weights = (-1.0) ** np.arange(n + 1)
    weights[[0, -1]] *= 0.5
    offsets = x[:, None] - x
    np.fill_diagonal(offsets, 1)
    derivative = weights[None, :] / weights[:, None] / offsets
    np.fill_diagonal(derivative, 0)
    derivative -= np.diag(derivative.sum(axis=1))
    return x, derivative


def traces(g, radius, orders):
    n = np.arange(orders + 2)
    z = g * radius
    i, k = ive(n, z), kve(n, z)
    return i[:-1], k[:-1], z * i[1:] / i[:-1] + n[:-1], n[:-1] - z * k[1:] / k[:-1]


class Exterior:
    def __init__(self, pe, m, radial, padding):
        self.m = m
        self.pe = pe
        self.k = pe / 2
        self.orders = np.arange(m[0] - padding, m[-1] + padding + 1)
        difference = self.orders[:, None] - self.orders
        self.a, self.angular = eigh(
            np.diag(self.orders.astype(float) ** 2) - self.k**2 * (np.abs(difference) == 2)
        )
        jump = self.orders[:, None] - m
        self.project = self.angular.T @ (iv(abs(jump), pe) * (-1.0) ** jump)
        self.unproject = iv(abs(jump.T), pe) @ self.angular
        x, self.dx = compact_grid(radial)
        self.x = x
        self.base = x[:, None] ** 2 * (self.dx @ self.dx) + 2 * x[:, None] * self.dx
        self.base += np.diag(0.25 - self.k**2 * x * x)
        self.stable = pe >= 5

    def physical_response(self, p, op):
        """Move the wall gauge inside the operator before taking its matrix function.

        In physical Fourier coordinates the angular operator becomes tridiagonal.
        A unitary radial Schur factorization avoids the exponential conditioning
        of the angular eigenvectors under this similarity transformation.
        """
        n = self.orders
        band = np.zeros((3, len(n)), dtype=complex)
        band[0, 1:] = -self.pe * (n[1:] - 0.5)
        band[2, :-1] = self.pe * (n[:-1] + 0.5)
        diag = n * n - self.pe**2 / 2
        unit = np.eye(len(n))[:, (self.m - self.orders[0])]
        T, Q = schur(op[:-1, :-1], output="complex", check_finite=False)
        forcing = -(Q.conj().T @ op[:-1, -1])
        dim = len(forcing)
        v = np.zeros((dim, len(n) * len(self.m)), dtype=complex)
        for j in range(dim - 1, -1, -1):
            rhs = forcing[j] * unit
            if j + 1 < dim:
                rhs -= (T[j, j + 1 :] @ v[j + 1 :]).reshape(unit.shape)
            band[1] = T[j, j] - diag
            v[j] = solve_banded((1, 1), band, rhs, check_finite=False).ravel()
        derivative = ((self.dx[-1, :-1] @ Q) @ v).reshape(unit.shape) + self.dx[-1, -1] * unit
        dtn = ((p + 0.5) * unit + derivative)[self.m - n[0]]
        return dtn, (Q, v, unit)

    def response(self, s, need_field=True):
        if self.pe == 0:
            g = np.sqrt(s)
            return np.diag(-traces(g, 1, self.m[-1])[3][abs(self.m)]), None
        p = np.sqrt(s + self.k**2)
        # R = exp(-p*(r-1))*U(1/r)/sqrt(r). At x=0 the regular
        # endpoint equation selects the decaying branch at infinity.
        op = self.base + 2 * p * self.dx
        if self.stable:
            return self.physical_response(p, op)
        size = len(self.x) - 1
        matrices = np.broadcast_to(op[:-1, :-1], (len(self.a), size, size)).copy()
        diag = np.arange(size)
        matrices[:, diag, diag] -= self.a[:, None]
        rhs = np.broadcast_to(-op[:-1, -1], (len(self.a), size))
        interior = np.linalg.solve(matrices, rhs[:, :, None])[:, :, 0]
        u = np.column_stack((interior, np.ones(len(interior))))
        z = p + 0.5 + u @ self.dx[-1]
        return (self.unproject * z) @ self.project, u


class BoundaryOperator:
    def __init__(self, case, pipe=16, wall=26, radial=30, padding=14):
        self.par = case
        self.n = np.arange(-pipe, pipe + 1)
        self.m = np.arange(-wall, wall + 1)
        self.ang = case["phi_v"] if case["Pe"] else 0.0
        self.centres = np.array(
            [complex(case[f"xp{i}"], case[f"yp{i}"]) * np.exp(-1j * self.ang) for i in (1, 2)]
        )
        self.radii = np.array([case[f"Rp{i}"] for i in (1, 2)])
        self.beta = np.array([case[f"beta{i}"] for i in (1, 2)])
        self.psi = np.array([case[f"psi{i}"] for i in (1, 2)])
        self.right = (1 + self.psi[1]) / self.psi.sum()
        self.exterior = Exterior(case["Pe"], self.m, radial, padding)
        self.cost = dict(radial=0.0, coupling=0.0, spatial=0.0)

    def frequency(self, s, need_wall):
        """Unit-load transfer (before division by s). Return the common fluid increment."""
        start = perf_counter()
        dtn, u = self.exterior.response(s, need_wall)
        stop = perf_counter()
        self.cost["radial"] += stop - start
        n, m = self.n, self.m
        g = np.sqrt(s * self.par["gamma"])
        ib, kb, di, dk = traces(g, 1, m[-1])
        pipe = [traces(g, r, n[-1]) for r in self.radii]
        outward, inward = [], []
        for r, d, (ip, kp, _, _) in zip(self.radii, self.centres, pipe):
            delta = m[:, None] - n
            shift = ive(np.arange(m[-1] + n[-1] + 1), g * abs(d))[abs(delta)]
            outward.append(
                kb[abs(m), None]
                / kp[abs(n)]
                * shift
                * np.exp(-1j * delta * np.angle(d))
                * np.exp(-g * (1 - r) + g.real * abs(d))
            )
            inward.append(
                (shift * np.exp(1j * delta * np.angle(d))).T
                * ip[abs(n), None]
                / ib[abs(m)]
                * np.exp(g.real * (r + abs(d) - 1))
            )
        out = np.hstack(outward)
        lam = self.par["lam"]
        # X maps both singular pipe expansions to the regular borehole expansion.
        X = solve(
            dtn + np.diag(lam * di[abs(m)]), -(dtn + np.diag(lam * dk[abs(m)])) @ out, check_finite=False
        )
        inc = np.vstack(inward) @ X
        count = len(n)
        for target, source in ((0, 1), (1, 0)):
            d = self.centres[target] - self.centres[source]
            delta = n - n[:, None]
            kdiff = kve(np.arange(2 * n[-1] + 1), g * abs(d))[abs(delta)]
            shift = (
                ((-1.0) ** n * pipe[target][0][abs(n)])[:, None]
                / pipe[source][1][abs(n)]
                * kdiff
                * np.exp(
                    1j * delta * np.angle(d) - g * (abs(d) - self.radii[source]) + g.real * self.radii[target]
                )
            )
            inc[target * count : (target + 1) * count, source * count : (source + 1) * count] += shift
        trace = inc + np.eye(2 * count)
        derivative = np.concatenate([v[2][abs(n)] for v in pipe])[:, None] * inc
        derivative += np.diag(np.concatenate([v[3][abs(n)] for v in pipe]))
        zero = np.array([n[-1], count + n[-1]])
        mean = self.psi @ trace[zero] / self.psi.sum()
        system = trace - np.repeat(self.beta / self.radii, count)[:, None] * derivative
        system[zero] -= mean
        rhs = np.zeros(2 * count)
        rhs[zero] = [self.right, self.right - 1]
        h = solve(system, rhs, check_finite=False)
        correction = mean @ h
        regular = X @ h
        wall = regular + out @ h
        self.cost["coupling"] += perf_counter() - stop
        return correction, (g, regular, h, wall, u, ib, pipe) if need_wall else None

    def profiles(self, s, state, grid):
        """Radial Fourier amplitudes on fixed probe-radius knots, before inversion."""
        start = perf_counter()
        g, w, h, b, u, ib, pipe = state
        grout = []
        for i, (radii, orders) in enumerate(zip(grid[:3], (self.m, self.n, self.n))):
            if i == 0:
                vals = ive(abs(orders), g * radii[:, None]) / ib[abs(orders)]
                vals *= np.exp(g.real * (radii - 1))[:, None] * w
            else:
                r0 = self.radii[i - 1]
                vals = kve(abs(orders), g * radii[:, None]) / pipe[i - 1][1][abs(orders)]
                vals *= np.exp(-g * (radii - r0))[:, None] * np.split(h, 2)[i - 1]
            grout.append(vals)
        ext = self.exterior
        r = grid[3]
        if ext.pe == 0:
            p = np.sqrt(s)
            vals = kve(abs(self.m), p * r[:, None]) / kve(abs(self.m), p)
            field = vals * np.exp(-p * (r - 1))[:, None] * b
        else:
            p = np.sqrt(s + ext.k**2)
            if ext.stable:
                Q, v, unit = u
                solved = v.reshape(len(Q), len(ext.orders), len(self.m)) @ b
                u = np.vstack((Q @ solved, unit @ b)).T
            if callable(u):
                vals = u(r)
            else:
                # Convert U values at ascending Chebyshev nodes in x to coefficients.
                coeff = dct(u[:, ::-1], type=1, axis=1) / (u.shape[1] - 1)
                coeff[:, [0, -1]] *= 0.5
                vals = np.polynomial.chebyshev.chebval(2 / r - 1, coeff.T).T
            field = vals if ext.stable else ((vals * (ext.project @ b)) @ ext.angular.T)
            field *= (np.exp(-(p - ext.k) * (r - 1) + ext.k) / np.sqrt(r))[:, None]
        self.cost["spatial"] += perf_counter() - start
        return [*grout, field]
