import numpy as np


def rectangular_probes(x, y):
    x_grid, y_grid = np.meshgrid(x, y)
    return np.column_stack((x_grid.ravel(), y_grid.ravel()))


def local_probes(case):
    axis = np.linspace(-3, 3, 121)
    xy = rectangular_probes(axis, axis)
    keep = np.ones(len(xy), bool)
    for k in (1, 2):
        keep &= np.hypot(xy[:, 0] - case[f"xp{k}"], xy[:, 1] - case[f"yp{k}"]) > case[f"Rp{k}"] + 1e-4
    return xy[keep]


def extended_probes(case):
    xy = rectangular_probes(np.linspace(-3, 25, 113), np.linspace(-8, 8, 65))
    xy = xy[np.linalg.norm(xy, axis=1) >= 1.0]
    c, s = np.cos(case["phi_v"]), np.sin(case["phi_v"])
    physical = xy @ np.array([[c, -s], [s, c]]).T
    return xy, physical


def make_solver(case, job):
    from .solver import GHEFEMSolver, NonDimParams

    solid = {key: case[key] for key in ("lam", "gamma", "beta1", "beta2", "psi1", "psi2", "Pe", "phi_v")}
    pipes = {key + "_star": case[key] for key in ("xp1", "yp1", "xp2", "yp2", "Rp1", "Rp2")}
    params = NonDimParams(**solid, **pipes, q0_hat=case["q0"], R_far_star=job["radius"])
    return GHEFEMSolver(nondim_params=params, mesh_refinement=job["mesh"])


def solve(model, time, load, snapshots=None):
    args = dict(time_grid=time, q_profile=load)
    if snapshots is not None:
        args["snapshot_times"] = snapshots
    result = model.solve_transient(float(time[-1]), float(time[1]), **args)
    checks = {key: float(result[key]) for key in ("max_discrete_residual", "max_coupling_defect")}
    if not all(np.isfinite(v) and v < 1e-9 for v in checks.values()):
        raise RuntimeError(f"The reference failed the discrete solve or fluid coupling check: {checks}")
    return result, checks


def field_samples(model, result, snapshots, xy):
    from dolfinx import geometry

    xyz = np.column_stack((xy, np.zeros(len(xy))))
    tree = geometry.bb_tree(model.msh, model.msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, xyz)
    collisions = geometry.compute_colliding_cells(model.msh, candidates, xyz)
    cells = np.array(
        [collisions.links(i)[0] if len(collisions.links(i)) else -1 for i in range(len(xy))], dtype=np.int32
    )
    if (cells < 0).any():
        raise ValueError("The mesh does not contain all of the common probes")
    values = []
    for tau in snapshots:
        model.theta.x.array[:] = result["snapshots"][tau]
        model.theta.x.scatter_forward()
        values.append(model.theta.eval(xyz, cells).reshape(-1))
    return np.asarray(values)


def fluid(result):
    return np.column_stack((result["T_f_in"], result["T_f_out"]))
