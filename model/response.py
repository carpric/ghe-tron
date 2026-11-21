"""N circular pipe multipoles with a specified linear fluid circuit for a unit load."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.interpolate import CubicSpline
from scipy.linalg import solve
from scipy.special import ive, kve

from .boundary import Exterior, traces
from .wall import WallTable
from .loads import reconstruct, uniform_grid

RULE = Path(__file__).resolve().parents[1] / "data/rules/fitted.npz"

FIELD_TIMES = np.array([0.02, 1.0, 10.0, 20.0, 100.0])
FIRST = 0.0005
ANCHORS = np.unique(
    np.r_[FIELD_TIMES, np.geomspace(FIRST, 100.0, int(np.ceil(6 * np.log10(100 / FIRST))) + 1)]
)


@dataclass(frozen=True)
class Circuit:
    """A F + B q_pipe = load_vector * q_total, q_pipe positive into the solids."""

    A: np.ndarray
    B: np.ndarray
    load_vector: np.ndarray
    branches: tuple = ()
    flow_fractions: tuple = ()

    def eliminate(self, psi):
        count = len(psi)
        A, B, b = (np.asarray(x, float) for x in (self.A, self.B, self.load_vector))
        if A.shape != (count, count) or B.shape != A.shape or b.shape != (count,):
            raise ValueError("Circuit must supply one independent equation per pipe")
        if not all(np.isfinite(x).all() for x in (A, B, b)):
            raise ValueError("Nonfinite circuit coefficients")
        conductance = B * psi
        solved = solve(A + conductance, np.column_stack((b, conductance)))
        right, wall = solved[:, 0], solved[:, 1:]
        if not (
            np.isclose(psi @ right, 1.0, atol=1e-11, rtol=0)
            and np.allclose(psi @ wall, psi, atol=1e-11, rtol=1e-11)
        ):
            raise ValueError("Circuit does not conserve the prescribed total load")
        return right, wall

    @classmethod
    def parallel_utubes(cls, pairs, fractions):
        """Each (supply leg, return leg) serves a fixed share of total mass flow."""
        pairs = np.asarray(pairs)
        f = np.asarray(fractions, float)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
            raise ValueError("Supply pairs of integer pipe indices")
        n = pairs.size
        if not n or not np.array_equal(np.sort(pairs.ravel()), np.arange(n)):
            raise ValueError("Each zero-based pipe index must appear once")
        if (
            f.shape != (len(pairs),)
            or not np.isfinite(f).all()
            or np.any(f <= 0)
            or not np.isclose(f.sum(), 1.0, atol=1e-12, rtol=0)
        ):
            raise ValueError("Positive branch fractions must sum to one")
        A = np.zeros((n, n))
        B = np.zeros_like(A)
        b = np.zeros(n)
        for j, ((s, r), frac) in enumerate(zip(pairs, f)):
            A[j, s], A[j, r] = -frac, frac
            B[j, [s, r]] = 1.0
        for j, s in enumerate(pairs[1:, 0], start=len(pairs)):
            A[j, s], A[j, pairs[0, 0]] = 1.0, -1.0
        B[-1] = 1.0
        b[-1] = 1.0
        return cls(A, B, b, tuple(map(tuple, pairs)), tuple(f))

    @classmethod
    def prescribed_heat_fractions(cls, fractions):
        """Independent pipes with specified fractions of one common heat input."""
        f = np.asarray(fractions, float)
        if (
            f.ndim != 1
            or not len(f)
            or not np.isfinite(f).all()
            or not np.isclose(f.sum(), 1.0, atol=1e-12, rtol=0)
        ):
            raise ValueError("Specified heat fractions must sum to one")
        return cls(np.zeros((len(f), len(f))), np.eye(len(f)), f)


class MultiPipeResponse:
    """Normalized pipe-fluid histories and grout/ground fields on physical Cartesian probes."""

    def __init__(
        self,
        centres,
        radii,
        beta,
        coupling,
        lam,
        gamma,
        pe,
        angle,
        circuit,
        table_directory=None,
        rule_file=RULE,
    ):
        xy = np.asarray(centres, float)
        self.radii, self.beta = (np.array(a, dtype=float, copy=True) for a in (radii, beta))
        self.count = len(self.radii)
        if (
            not self.count
            or xy.shape != (self.count, 2)
            or self.beta.shape != (self.count,)
            or self.radii.shape != (self.count,)
        ):
            raise ValueError("Supply a centre, radius and wall coefficient for each pipe")
        if not all(
            np.isfinite(a).all() for a in (xy, self.radii, self.beta, [coupling, lam, gamma, pe, angle])
        ):
            raise ValueError("Nonfinite exchanger inputs")
        if min(self.radii.min(), self.beta.min(), coupling, lam, gamma) <= 0 or not 0 <= pe <= 10:
            raise ValueError("Positive thermal parameters and 0 <= Pe <= 10 required")
        if np.any(np.linalg.norm(xy, axis=1) + self.radii >= 1.0):
            raise ValueError("Pipe touching or outside the borehole")
        gap = np.linalg.norm(xy[:, None] - xy, axis=2) - (self.radii[:, None] + self.radii)
        if np.any(gap[np.triu_indices(self.count, 1)] <= 0):
            raise ValueError("Overlapping or touching pipes")
        self.ang = angle if pe else 0.0
        self.centres = (xy[:, 0] + 1j * xy[:, 1]) * np.exp(-1j * self.ang)
        self.psi = coupling * self.radii / self.beta
        self.circuit = circuit
        self.right, self.wall_to_fluid = circuit.eliminate(self.psi)
        self.lam, self.gamma = lam, gamma
        self.n, self.m = np.arange(-12, 13), np.arange(-22, 23)
        self.par = {"Pe": pe}
        self.exterior = Exterior(pe, self.m, 28, 14)
        if pe and table_directory is not None:
            WallTable(table_directory, 22, 28, 14).attach(self)
        with np.load(rule_file) as w:
            self.poles, self.weights = w["z"], w["w"]
        if (
            self.poles.ndim != 1
            or self.poles.size == 0
            or self.weights.shape != self.poles.shape
            or not np.isfinite(self.poles).all()
            or not np.isfinite(self.weights).all()
        ):
            raise ValueError("Supply matching finite vectors of inversion poles and residues")
        self._fluid_spline = None

    def frequency(self, s, need_field=False):
        """Return dynamic per-pipe fluid increments and, optionally, spatial coefficients."""
        dtn, radial = self.exterior.response(s, need_field)
        n, m = self.n, self.m
        g = np.sqrt(s * self.gamma)
        ib, kb, di, dk = traces(g, 1, m[-1])
        pipe = [traces(g, r, n[-1]) for r in self.radii]
        outward, inward = [], []
        delta = m[:, None] - n
        for r, c, (ip, kp, _, _) in zip(self.radii, self.centres, pipe):
            shift = ive(np.arange(m[-1] + n[-1] + 1), g * abs(c))[abs(delta)]
            outward.append(
                kb[abs(m), None]
                / kp[abs(n)]
                * shift
                * np.exp(-1j * delta * np.angle(c))
                * np.exp(-g * (1 - r) + g.real * abs(c))
            )
            inward.append(
                (shift * np.exp(1j * delta * np.angle(c))).T
                * ip[abs(n), None]
                / ib[abs(m)]
                * np.exp(g.real * (r + abs(c) - 1))
            )
        out = np.hstack(outward)
        X = solve(
            dtn + np.diag(self.lam * di[abs(m)]),
            -(dtn + np.diag(self.lam * dk[abs(m)])) @ out,
            check_finite=False,
        )
        width = len(n)
        incident = np.vstack(inward) @ X
        blocks = incident.reshape(self.count, width, self.count, width)
        delta = n - n[:, None]
        for i in range(self.count):
            for j in range(self.count):
                if i == j:
                    continue
                d = self.centres[i] - self.centres[j]
                kdiff = kve(np.arange(2 * n[-1] + 1), g * abs(d))[abs(delta)]
                blocks[i, :, j, :] += (
                    ((-1.0) ** n * pipe[i][0][abs(n)])[:, None]
                    / pipe[j][1][abs(n)]
                    * kdiff
                    * np.exp(1j * delta * np.angle(d) - g * (abs(d) - self.radii[j]) + g.real * self.radii[i])
                )
        trace = incident + np.eye(self.count * width)
        derivative = np.concatenate([p[2][abs(n)] for p in pipe])[:, None] * incident
        derivative += np.diag(np.concatenate([p[3][abs(n)] for p in pipe]))
        zero = np.arange(self.count) * width + n[-1]
        increment = self.wall_to_fluid @ trace[zero]
        system = trace - np.repeat(self.beta / self.radii, width)[:, None] * derivative
        system[zero] -= increment
        rhs = np.zeros(len(trace))
        rhs[zero] = self.right
        h = solve(system, rhs, check_finite=False)
        regular = X @ h
        state = (g, regular, h.reshape(self.count, width), regular + out @ h, radial, ib, pipe)
        return increment @ h, state if need_field else None

    def _fluid_at(self, t, exact_wall=False):
        change = np.zeros(self.count)
        for z, w in zip(self.poles, self.weights):
            inc, _ = self.frequency(z / t, exact_wall)
            change += (w * inc).real
        return self.right + change

    def fluid(self, times):
        t = np.asarray(times, float)
        if t.ndim != 1 or not np.isfinite(t).all() or np.any((t < 0) | (t > 100)):
            raise ValueError("Provide a vector of times in [0,100]")
        if self._fluid_spline is None:
            anchors = [self._fluid_at(x, x in FIELD_TIMES) for x in ANCHORS]
            self._fluid_spline = CubicSpline(np.log(ANCHORS), anchors, axis=0)
        result = np.tile(self.right, (len(t), 1))
        positive = t >= FIRST
        result[positive] = self._fluid_spline(np.log(t[positive]))
        for i in np.flatnonzero((t > 0) & ~positive):
            result[i] = self._fluid_at(t[i])
        if not np.isfinite(result).all():
            raise ArithmeticError("Fluid response is outside numerical resolution")
        return result

    def signed_fluid(self, time, load):
        t = uniform_grid(time)
        return reconstruct(self.fluid(t), load)

    def circuit_temperatures(self, time, load=None):
        """Parallel U-tubes: common supply, branch returns and flow-weighted mixed return.

        Omit load for the unit step (right limit at zero). A load vector specifies
        the signed heat input on successive uniform intervals after a zero pre-load.
        """
        if not self.circuit.branches:
            raise ValueError("This readout requires a parallel U-tube circuit")
        legs = self.fluid(time) if load is None else self.signed_fluid(time, load)
        pairs = np.asarray(self.circuit.branches)
        returns = legs[:, pairs[:, 1]]
        return {
            "supply": legs[:, pairs[0, 0]],
            "returns": returns,
            "mixed_return": returns @ self.circuit.flow_fractions,
        }

    def fields(self, t, points):
        """Evaluate an arbitrary set of grout and unbounded-ground points at one unit-step time."""
        if not np.isscalar(t) or not np.isfinite(t) or not 0 <= t <= 100:
            raise ValueError("Snapshot time must belong to [0,100]")
        xy = np.asarray(points, float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
            raise ValueError("Supply a nonempty array of Cartesian probe points")
        z = (xy[:, 0] + 1j * xy[:, 1]) * np.exp(-1j * self.ang)
        if np.any(abs(z[:, None] - self.centres) < self.radii - 1e-12):
            raise ValueError("A solid probe lies inside a pipe")
        if t == 0:
            return np.zeros(len(xy))
        # Preserve the frozen wall-side convention on probes exactly at unit radius.
        grout = abs(z) < 1.0
        inner = z[grout]
        locations = [inner, *[inner - c for c in self.centres], z[~grout]]
        bounds = [
            (0.0, 1.0),
            *zip(self.radii, 1 + abs(self.centres)),
            (1.0, max(1.01, np.max(abs(z[~grout]), initial=1.0))),
        ]
        orders = [self.m, *([self.n] * self.count), self.exterior.orders if self.exterior.pe else self.m]
        grids = [a + (b - a) * np.linspace(0.0, 1.0, 97) ** 2 for a, b in bounds]
        phases = [np.exp(1j * np.angle(v)[:, None] * o) for v, o in zip(locations, orders)]
        e = self.exterior
        v = locations[-1]
        r = abs(v)
        exponent = e.k * (v.real + v.real / r**2 - r)
        if e.stable:
            exponent -= e.pe * v.real / r
        phases[-1] *= np.exp(exponent)[:, None]
        totals = [np.zeros((len(grid), len(order)), complex) for grid, order in zip(grids, orders)]
        for pole, w in zip(self.poles, self.weights):
            s = pole / t
            _, (g, regular, h, wall, radial, ib, pipe) = self.frequency(s, True)
            r = grids[0]
            totals[0] += (
                w
                * ive(abs(self.m), g * r[:, None])
                / ib[abs(self.m)]
                * np.exp(g.real * (r - 1))[:, None]
                * regular
            )
            for i, r in enumerate(grids[1:-1]):
                totals[i + 1] += (
                    w
                    * kve(abs(self.n), g * r[:, None])
                    / pipe[i][1][abs(self.n)]
                    * np.exp(-g * (r - self.radii[i]))[:, None]
                    * h[i]
                )
            r = grids[-1]
            if e.pe:
                p = np.sqrt(s + e.k**2)
                if e.stable:
                    Q, v, unit = radial
                    solved = v.reshape(len(Q), len(e.orders), len(self.m)) @ wall
                    radial = np.vstack((Q @ solved, unit @ wall)).T
                coeff = dct(radial[:, ::-1], type=1, axis=1) / (radial.shape[1] - 1)
                coeff[:, [0, -1]] *= 0.5
                amplitudes = np.polynomial.chebyshev.chebval(2 / r - 1, coeff.T).T
                if not e.stable:
                    amplitudes = (amplitudes * (e.project @ wall)) @ e.angular.T
                profile = amplitudes * (np.exp(-(p - e.k) * (r - 1) + e.k) / np.sqrt(r))[:, None]
            else:
                p = np.sqrt(s)
                profile = (
                    kve(abs(self.m), p * r[:, None])
                    / kve(abs(self.m), p)
                    * np.exp(-p * (r - 1))[:, None]
                    * wall
                )
            totals[-1] += w * profile
        parts = [
            np.einsum("ij,ij->i", CubicSpline(grid, total, axis=0)(abs(v)), phase).real
            for grid, total, v, phase in zip(grids, totals, locations, phases)
        ]
        result = np.empty(len(xy))
        result[grout] = sum(parts[:-1])
        result[~grout] = parts[-1]
        if not np.isfinite(result).all():
            raise ArithmeticError("Nonfinite field reconstruction")
        return result
