"""Joint single-U fluid and field evaluation on the published grids."""

import numpy as np
from scipy.interpolate import CubicSpline
from model.boundary import BoundaryOperator
from studies.inputs import probes, time_nodes, GRID, RULES

TIMES = np.array(GRID["snapshots"][1:])


def read_rule(filename="fitted.npz"):
    with np.load(RULES / filename) as data:
        return data["z"], data["w"]


class Readout:
    def __init__(self, model, k, locations=None):
        local, downstream = probes(model.par) if locations is None else locations
        xy = (local[:, 0] + 1j * local[:, 1]) * np.exp(-1j * model.ang)
        self.grout = abs(xy) < 1
        self.nlocal = len(xy)
        self.downstream = downstream
        if model.par["Pe"]:
            xy = np.r_[xy, (downstream[:, 0] + 1j * downstream[:, 1]) * np.exp(-1j * model.ang)]
        self.region = [
            xy[: self.nlocal][self.grout],
            xy[: self.nlocal][self.grout] - model.centres[0],
            xy[: self.nlocal][self.grout] - model.centres[1],
            np.r_[xy[: self.nlocal][~self.grout], xy[self.nlocal :]],
        ]
        mins = (0.0, *model.radii, 1.0)
        maxs = (1.0, *(1 + abs(model.centres)), np.max(abs(self.region[-1])))
        self.grid = [lo + (hi - lo) * np.linspace(0, 1, k + 1) ** 2 for lo, hi in zip(mins, maxs)]
        self.orders = (model.m, model.n, model.n, model.exterior.orders if model.par["Pe"] else model.m)
        self.phases = [
            np.exp(1j * np.angle(z)[:, None] * order) for z, order in zip(self.region, self.orders)
        ]
        z = self.region[-1]
        # Bound the physical angular exponent even at the farthest probes.
        exponent = model.exterior.k * (z.real + z.real / abs(z) ** 2 - abs(z))
        if model.exterior.stable:
            exponent -= model.par["Pe"] * z.real / abs(z)
        self.phases[-1] *= np.exp(exponent)[:, None]

    def field(self, coeff):
        values = []
        for z, nodes_r, phase, c in zip(self.region, self.grid, self.phases, coeff):
            vals = CubicSpline(nodes_r, c, axis=0)(abs(z))
            values.append(np.einsum("ij,ij->i", vals, phase).real)
        local = np.empty(self.nlocal)
        local[self.grout] = sum(values[:3])
        ng = np.sum(~self.grout)
        local[~self.grout] = values[-1][:ng]
        return local, values[-1][ng:]


def predict(case, rule, table=None, locations=None):
    model = BoundaryOperator(case, 12, 22, 28, 14)
    if table is not None and case["Pe"] > 0:
        table.attach(model)
    read = Readout(model, 96, locations)
    time = time_nodes(2)
    anchors = np.unique(
        np.r_[TIMES, np.geomspace(time[1], 100, int(np.ceil(6 * np.log10(100 / time[1]))) + 1)]
    )
    means, local, wake = [], [], []
    for t in anchors:
        field = t in TIMES
        sums = None
        mean = 0.0
        z, weights = rule
        for freq, weight in zip(z / t, weights):
            value, state = model.frequency(freq, field)
            mean += (weight * value).real
            if field:
                v = model.profiles(freq, state, read.grid)
                if sums is None:
                    sums = [weight * part for part in v]
                else:
                    for out, part in zip(sums, v):
                        out += weight * part
        means.append(mean)
        if field:
            loc, down = read.field(sums)
            local.append(loc)
            if t in (10, 20, 100):
                wake.append(down)
    change = CubicSpline(np.log(anchors), means)(np.log(time[1:]))
    inlet = model.right + np.r_[0.0, change]
    ds = dict(
        tau=time,
        fluid=np.column_stack((inlet, inlet - 1)),
        local=np.vstack([np.zeros_like(local[0]), local]),
        wake=np.array(wake),
        grout=read.grout,
    )
    if not all(np.isfinite(x).all() for x in ds.values()):
        raise ArithmeticError("Nonfinite response")
    return ds


def rows_for(ref, other, case):
    time = ref["tau"]
    for j, quantity in enumerate(("inlet", "outlet")):
        vals = np.interp(time, other["tau"], other["fluid"][:, j])
        for cutoff in (0.02, 2.0, 100.0):
            inside = time <= cutoff

            def norm(v):
                return np.sqrt(np.trapezoid(v[inside] ** 2, time[inside]) / cutoff)

            yield quantity, cutoff, norm(vals - ref["fluid"][:, j]), norm(ref["fluid"][:, j])
    local, _ = probes(case)
    for quantity, mask in (
        ("grout", np.linalg.norm(local, axis=1) < 1),
        ("ground", np.linalg.norm(local, axis=1) >= 1),
    ):
        for j, t in enumerate(TIMES, 1):
            v = ref["local"][j, mask]
            yield quantity, t, np.sqrt(np.mean((other["local"][j, mask] - v) ** 2)), np.sqrt(np.mean(v**2))
    if case["Pe"]:
        for j, t in enumerate((10.0, 20.0, 100.0)):
            v = ref["wake"][j]
            yield "wake", t, np.sqrt(np.mean((other["wake"][j] - v) ** 2)), np.sqrt(np.mean(v**2))
