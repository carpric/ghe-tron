"""Independent P1 finite-element reference for the 90-degree Rico--Hermanns case.

No boundary-response solver, model weights, published temperature values, or
fitted wall impedances enter these spatial or temporal solves.
"""

import argparse
import csv
import numpy as np
from scipy.sparse import bmat, csc_matrix, diags
from scipy.sparse.linalg import splu

from studies.inputs import RESULTS
from fea.parallel import assemble, mesh

HERE = RESULTS / "asymptotic"

PE = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
RADIUS = np.array([20.0, 25.0]) / 76.0
ANGLE = np.deg2rad([135.0, 45.0])
CENTRES = np.array([51.0, 46.0])[:, None] * np.c_[np.cos(ANGLE), np.sin(ANGLE)] / 76.0
BETA = 1.50 * np.array([0.535, 0.544]) * RADIUS
POWER = np.array([5.0, 1.0])  # q_c per unit length, temperature unit q_c/k_ground


def extend(base, pe, level):
    """Nest two outer boundaries around the identical Gmsh near-borehole mesh."""
    xy, triangles, ng, far, walls = base
    phi = np.arctan2(xy[far, 1], xy[far, 0])
    order = np.argsort(phi)
    prev = far[order]
    unit = np.c_[np.cos(phi[order]), np.sin(phi[order])]
    n = len(far)
    coords, cells = [xy], [triangles]
    start, count = 50.0, len(xy)
    r1 = max(60.0, 16.0 / pe)
    # Cap radial element Pe and grade the outer spacing with distance.
    for target in (r1, 2 * r1):
        while start < target:
            spacing = min(0.04 * start / level, 1.5 / pe)
            intervals = max(1, int(np.ceil((target - start) / spacing)))
            r = target if intervals == 1 else start + (target - start) / intervals
            new = np.arange(count, count + n)
            coords.append(r * unit)
            cells.extend(
                [np.c_[prev, new, np.roll(new, -1)], np.c_[prev, np.roll(new, -1), np.roll(prev, -1)]]
            )
            start, count, prev = r, count + n, new
    return (np.vstack(coords), np.vstack(cells), ng, prev, walls), r1


def blocks(mesh_data, pe):
    M, K, H, W, keep = assemble(mesh_data, pe, lam=0.5, gamma=2.24, angle=0.0, wall_beta=BETA)
    # H_ki is the integrated pipe Robin conductance times shape function k.
    # The polygon perimeter is used consistently in the domain and pipe rows.
    h = H.sum(axis=0)
    np.testing.assert_allclose(W, H.T / h[:, None], atol=1e-14)
    return M.tocsc(), K.tocsc(), csc_matrix(H), h, keep


def system(K, H, h, M=None, a=0.0):
    top = K if M is None else K + a * M
    return bmat([[top, -H], [-H.T, diags(h)]], format="csc")


def inspect_solution(H, h, state):
    temperatures = state[-2:]
    supplied = h * temperatures - H.T @ state[:-2]
    defect = np.max(abs(supplied - POWER))
    if not np.all(np.isfinite(state)) or defect > 1e-8:
        raise ArithmeticError(f"Invalid FE solution, pipe-power defect {defect:g}")
    return temperatures, defect


def steady(K, H, h):
    A = system(K, H, h)
    rhs = np.r_[np.zeros(K.shape[0]), POWER]
    state = splu(A).solve(rhs)
    residual = np.max(abs(A @ state - rhs))
    temp, defect = inspect_solution(H, h, state)
    if residual > 1e-8:
        raise ArithmeticError(f"Steady FE residual {residual:g}")
    return temp, defect


def transient(M, K, H, h, dt):
    # Two initial backward-Euler half steps damp the step discontinuity.
    # Subsequent Crank--Nicolson steps use the same block factorization.
    alpha = 2.0 / dt
    lu = splu(system(K, H, h, M, alpha))
    state = np.r_[np.zeros(K.shape[0]), POWER / h]
    records = []
    for _ in range(2):
        state = lu.solve(np.r_[alpha * (M @ state[:-2]), POWER])
    inspect_solution(H, h, state)
    rhs_op = alpha * M - K
    for step in range(2, round(100 / dt) + 1):
        state = lu.solve(np.r_[rhs_op @ state[:-2] + H @ state[-2:], POWER])
        t = step * dt
        if t >= 10 and t.is_integer():
            temp, defect = inspect_solution(H, h, state)
            records.append((t, *temp, defect))
    return records


def write(file, rows):
    with file.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def solve_level(level, kind):
    HERE.mkdir(exist_ok=True, parents=True)
    base = mesh(level, CENTRES, RADIUS)
    records = []
    for pe in PE if kind == "steady" else (0.1, 0.5):
        grid, r1 = extend(base, pe, level)
        M, K, H, h, keep = blocks(grid, pe)
        for radius in (r1, 2 * r1):
            # Rows exactly on the inner candidate boundary carry zero Dirichlet data.
            mask = np.linalg.norm(grid[0][keep], axis=1) < radius * (1 - 1e-10)
            Kr, Hr = K[mask][:, mask], H[mask]
            common = dict(pe=pe, level=level, radius=radius, solid_dofs=Kr.shape[0])
            if kind == "steady":
                temp, defect = steady(Kr, Hr, h)
                records.append(dict(**common, pipe1=temp[0], pipe2=temp[1], power_defect=defect))
            else:
                Mr = M[mask][:, mask]
                steps = (0.5,) if level == 1 else (0.5, 0.25) if radius == 2 * r1 and level == 2 else (0.25,)
                for dt in steps:
                    for t, t1, t2, defect in transient(Mr, Kr, Hr, h, dt):
                        records.append(dict(**common, dt=dt, tau=t, pipe1=t1, pipe2=t2, power_defect=defect))
        print(f"{kind} level {level} Pe {pe:g}, {len(keep)} solid DOFs", flush=True)
    write(HERE / f"fea_{kind}_L{level}.csv", records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["steady", "transient"])
    parser.add_argument("level", type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    solve_level(args.level, args.kind)
