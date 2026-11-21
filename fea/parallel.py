"""Independent P1 Galerkin / backward Euler reference, with explicit branch balances.

Only this assessment uses a mesh. All pipe temperatures are solved at the current
time step together with the plane temperature. No boundary-operator code is used.
"""

import gmsh
import numpy as np
from scipy.linalg import solve
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu
from matplotlib.tri import Triangulation
from studies.circuits import (
    ANGLE,
    BETA,
    CASES,
    CENTRES,
    COUPLING,
    GAMMA,
    HERE,
    LAMBDA,
    PAIRS,
    RADII,
    SNAP,
    name,
    probes,
    signed_command,
    times,
)


def mesh(level, centres=CENTRES, radii=RADII):
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("parallel_utubes")
        geo = gmsh.model.geo

        def circle(x, y, r, size):
            mid = geo.addPoint(x, y, 0.0, size)
            sides = [
                geo.addPoint(x + r * cx, y + r * cy, 0.0, size)
                for cx, cy in [(1, 0), (0, 1), (-1, 0), (0, -1)]
            ]
            arcs = [geo.addCircleArc(p, mid, sides[(i + 1) % 4]) for i, p in enumerate(sides)]
            return geo.addCurveLoop(arcs), arcs

        inner, _ = circle(0.0, 0.0, 1.0, 0.10 / level)
        outer, far = circle(0.0, 0.0, 50.0, 2.0 / level)
        pipes = [circle(x, y, r, 0.04 / level) for (x, y), r in zip(centres, radii)]
        regions = [geo.addPlaneSurface([inner, *[p[0] for p in pipes]]), geo.addPlaneSurface([outer, inner])]
        geo.synchronize()
        gmsh.model.mesh.generate(2)

        def elements(dim, tag, etype):
            types, _, nodes = gmsh.model.mesh.getElements(dim, tag)
            return nodes[list(types).index(etype)].reshape(-1, dim + 1)

        groups = [elements(2, r, 2) for r in regions]
        tri = np.vstack(groups)
        used = np.unique(tri)
        tag, coords, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tag)
        xy = coords.reshape(-1, 3)[order[np.searchsorted(tag[order], used)], :2]

        def edge_ids(arcs):
            return np.searchsorted(used, np.vstack([elements(1, a, 1) for a in arcs]))

        return (
            xy,
            np.searchsorted(used, tri),
            len(groups[0]),
            np.unique(edge_ids(far)),
            [edge_ids(p[1]) for p in pipes],
        )
    finally:
        gmsh.finalize()


def assemble(grid, pe, lam=LAMBDA, gamma=GAMMA, angle=ANGLE, wall_beta=BETA):
    xy, tri, ng, outer, walls = grid
    p = xy[tri]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    area = np.abs(det) / 2.0
    edges = p[:, [2, 0, 1]] - p[:, [1, 2, 0]]
    grad = np.stack([-edges[:, :, 1], edges[:, :, 0]], axis=2) / det[:, None, None]
    cond = np.ones(len(p))
    cond[:ng] = lam
    capacity = np.ones(len(p))
    capacity[:ng] = lam * gamma
    mass = (capacity * area / 12)[:, None, None] * (np.ones((3, 3)) + np.eye(3))
    stiffness = (cond * area)[:, None, None] * (grad @ grad.transpose(0, 2, 1))
    # Three Gauss points in each ground triangle for v * (u dot grad T).
    rot = np.exp(1j * angle)
    for bary in np.eye(3) / 2 + 1 / 6:
        x, y = (bary @ p[ng:]).T
        z = (x + 1j * y) / rot
        vel = pe * rot * (1 - 1 / np.conj(z) ** 2)
        adv = np.einsum("tik,tk->ti", grad[ng:], np.c_[vel.real, vel.imag])
        stiffness[ng:] += (area[ng:] / 3)[:, None, None] * bary[None, :, None] * adv[:, None, :]

    def matrix(local):
        return coo_matrix(
            (local.ravel(), (np.repeat(tri, 3, axis=1).ravel(), np.tile(tri, (1, 3)).ravel())),
            shape=(len(xy), len(xy)),
        ).tocsr()

    M, K = matrix(mass), matrix(stiffness)
    pipe_rhs, averages = [], []
    for e, beta in zip(walls, wall_beta):
        length = np.linalg.norm(xy[e[:, 0]] - xy[e[:, 1]], axis=1)
        integ = np.bincount(e.ravel(), weights=np.repeat(length / 2, 2), minlength=len(xy))
        averages.append(integ / length.sum())
        pipe_rhs.append(lam / beta * integ)
        local = (length * lam / (6 * beta))[:, None, None] * np.array([[2, 1], [1, 2]])
        K += coo_matrix(
            (local.ravel(), (np.repeat(e, 2, axis=1).ravel(), np.tile(e, (1, 2)).ravel())), shape=K.shape
        ).tocsr()
    keep = np.setdiff1d(np.arange(len(xy)), outer)
    return M[keep][:, keep], K[keep][:, keep], np.array(pipe_rhs).T[keep], np.array(averages)[:, keep], keep


def observations(grid, keep, locations=None):
    xy, tri, *_ = grid
    local, wake = probes() if locations is None else locations
    points = np.vstack([local, wake])
    triangles = Triangulation(xy[:, 0], xy[:, 1], tri)
    idx = triangles.get_trifinder()(*points.T)
    if np.any(idx < 0):
        raise ValueError("FE mesh misses an observation point")
    verts = tri[idx]
    p = xy[verts]
    coeff = np.linalg.solve((p[:, 1:] - p[:, :1]).transpose(0, 2, 1), (points - p[:, 0])[:, :, None])[:, :, 0]
    bary = np.c_[1 - coeff.sum(axis=1), coeff]
    return csr_matrix(
        (bary.ravel(), (np.repeat(np.arange(len(points)), 3), verts.ravel())), shape=(len(points), len(xy))
    )[:, keep]


def circuit(pairs, fractions, psi, averages, pipe_rhs):
    f = np.asarray(fractions, float)
    nbranch = len(pairs)
    # z = [common supply, return_1, ..., return_B]. Bottom row sets the total load.
    leg = np.zeros((len(psi), 1 + nbranch))
    solid = np.zeros((nbranch + 1, averages.shape[1]))
    network = np.zeros((nbranch + 1, nbranch + 1))
    for b, (i, j) in enumerate(pairs):
        leg[i, 0] = leg[j, b + 1] = 1
        solid[b] = -psi[i] * averages[i] - psi[j] * averages[j]
        network[b, 0], network[b, b + 1] = psi[i] - f[b], psi[j] + f[b]
    network[-1] = np.r_[1, -f]
    return csr_matrix(solid), network, pipe_rhs @ leg, f


def integrate(M, K, systems, read, level, signed):
    nbranch = len(systems[0][-1])
    count = len(systems)
    time = np.linspace(0.0, 20.0, 200 * level + 1) if signed else times(level)
    load = signed_command(time) if signed else np.ones_like(time)
    unit = np.r_[np.zeros(nbranch), 1.0]
    theta = np.zeros((M.shape[0], count))
    fluids = np.empty((count, len(time), nbranch + 2))
    snapshots = np.array([]) if signed else SNAP
    fields = np.full((count, len(snapshots), 0 if signed else read.shape[0]), np.nan)
    for k, (_, net, _, f) in enumerate(systems):
        z = solve(net, unit * load[0])
        fluids[k, 0] = np.r_[z, f @ z[1:]]
    prev = None
    for it in range(1, len(time)):
        dt = round(time[it] - time[it - 1], 10)
        if dt != prev:
            lu = splu((M / dt + K).tocsc())
            gains = [lu.solve(H) for _, _, H, _ in systems]
            schur = [net + G @ Y for (G, net, _, _), Y in zip(systems, gains)]
            prev = dt
        free = lu.solve(M @ theta / dt)
        for k, ((G, _, _, f), Y, S) in enumerate(zip(systems, gains, schur)):
            z = solve(S, unit * load[it] - G @ free[:, k])
            theta[:, k] = free[:, k] + Y @ z
            fluids[k, it] = np.r_[z, f @ z[1:]]
        idx = np.flatnonzero(np.isclose(snapshots, time[it], atol=1e-10, rtol=0))
        if len(idx):
            fields[:, idx[0]] = (read @ theta).T
    for solution in fluids:
        np.testing.assert_allclose(solution[:, 0] - solution[:, -1], load, atol=2e-12, rtol=0)
    if not np.isfinite(fields).all():
        raise ArithmeticError("Nonfinite FE field")
    return time, fluids, fields


def run(pe, level, signed=False):
    grid = mesh(level)
    M, K, pipe_rhs, averages, keep = assemble(grid, pe)
    read = observations(grid, keep) if not signed else None
    cases = [("opposite", 0.35)] if signed else [(k, f) for k, v, f in CASES if v == pe]
    systems = [circuit(PAIRS[k], [f, 1 - f], COUPLING * RADII / BETA, averages, pipe_rhs) for k, f in cases]
    time, fluids, fields = integrate(M, K, systems, read, level, signed)
    dest = HERE / "raw"
    dest.mkdir(exist_ok=True)
    for i, (k, f) in enumerate(cases):
        prefix = "signed_" if signed else ""
        path = dest / f"{prefix}{name(k,pe,f)}_fe{level}.npz"
        np.savez_compressed(path, time=time, fluid=fluids[i], field=fields[i], ndof=len(keep))
    print("FE complete", pe, level, len(keep), flush=True)
