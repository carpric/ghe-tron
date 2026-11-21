"""Reproduce the fixed asymmetric parallel-U study (predict, reference, score)."""

import json

import numpy as np

from studies.inputs import DATA
from studies.circuits import HERE, SNAP, probes, signed_command, times

PANEL = HERE / "confirmation"
RAW = PANEL / "raw"


def load_cases():
    return json.loads((DATA / "parallel_u.json").read_text())


def locations(case):
    return probes(np.array(case["centres"]), np.array(case["radii"]), np.deg2rad(case["flow_angle_deg"]))


def path(case, method, signed=False):
    suffix = "_signed" if signed else ""
    return RAW / f"{case['id']}{suffix}_{method}.npz"


def ports(op, time, load=None):
    temperatures = op.circuit_temperatures(time, load)
    matrix = np.column_stack([temperatures["supply"], temperatures["returns"], temperatures["mixed_return"]])
    np.testing.assert_allclose(
        matrix[:, 0] - matrix[:, -1], 1.0 if load is None else load, rtol=0, atol=5e-11
    )
    return matrix


def predict(case):
    from model.response import Circuit, MultiPipeResponse

    op = MultiPipeResponse(
        case["centres"],
        case["radii"],
        case["beta"],
        case["coupling"],
        case["lam"],
        case["gamma"],
        case["pe"],
        np.deg2rad(case["flow_angle_deg"]),
        Circuit.parallel_utubes(case["pairs"], case["fractions"]),
        HERE / "raw/tables",
    )
    time = times(2)
    points = np.vstack(locations(case))
    field = np.array([op.fields(tau, points) for tau in SNAP])
    np.savez_compressed(path(case, "op"), time=time, fluid=ports(op, time), field=field)
    if case["signed"]:
        time = np.linspace(0.0, 20.0, 401)
        np.savez_compressed(path(case, "op", True), time=time, fluid=ports(op, time, signed_command(time)))
    print("Predicted", case["id"], flush=True)


def reference(job):
    from fea import parallel as fe

    case, level = job
    grid = fe.mesh(level, case["centres"], case["radii"])
    mass, stiff, rhs, average, keep = fe.assemble(
        grid, case["pe"], case["lam"], case["gamma"], np.deg2rad(case["flow_angle_deg"]), case["beta"]
    )
    psi = case["coupling"] * np.array(case["radii"]) / case["beta"]
    system = fe.circuit(case["pairs"], case["fractions"], psi, average, rhs)
    probes_fe = fe.observations(grid, keep, locations(case))
    for signed in [False, True] if case["signed"] else [False]:
        time, fluids, fields = fe.integrate(mass, stiff, [system], probes_fe, level, signed)
        np.savez_compressed(
            path(case, f"fe{level}", signed), time=time, fluid=fluids[0], field=fields[0], ndof=len(keep)
        )
    print("FE complete", case["id"], level, len(keep), flush=True)


def scores(case, signed, fine_level=2):
    rows = []
    local, wake = locations(case)
    grout = np.linalg.norm(local, axis=1) < 1
    with (
        np.load(path(case, "op", signed)) as op,
        np.load(path(case, f"fe{fine_level}", signed)) as fine,
        np.load(path(case, f"fe{fine_level//2}", signed)) as coarse,
    ):
        # Additional third-level comparisons use the common original fine nodes.
        stride = fine_level // 2
        t = fine["time"][::stride]
        true_fluid = fine["fluid"][::stride]
        np.testing.assert_allclose(t, op["time"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(fine["time"][::2], coarse["time"], rtol=0, atol=1e-12)

        def record(name, tau, discrepancy, paired, signal, coarse_signal):
            rows.append(
                dict(
                    case=case["id"],
                    pipes=len(case["radii"]),
                    pe=case["pe"],
                    forcing="signed" if signed else "step",
                    quantity=name,
                    tau=tau,
                    rms=discrepancy,
                    reference_rms=signal,
                    relative_pct=100 * discrepancy / signal,
                    coarse_fine_rms=paired,
                    coarse_fine_pct=100 * paired / coarse_signal,
                    coarse_dof=int(coarse["ndof"]),
                    fine_dof=int(fine["ndof"]),
                )
            )

        names = ["supply", *[f"return_{b+1}" for b in range(len(case["pairs"]))], "mixed_return"]
        for j, name in enumerate(names):
            true = true_fluid[:, j]
            for horizon in [20.0] if signed else [0.02, 2.0, 100.0]:
                mask = t <= horizon

                def norm(values, nodes):
                    return np.sqrt(np.trapezoid(values**2, nodes) / horizon)

                signal = norm(true[mask], t[mask])
                error = norm((op["fluid"][:, j] - true)[mask], t[mask])
                if signed:
                    # Compare BE endpoints at common times, preserving exact load jumps.
                    nodes = coarse["time"]
                    delta = norm(coarse["fluid"][:, j] - fine["fluid"][::2, j], nodes)
                    coarse_signal = norm(fine["fluid"][::2, j], nodes)
                else:
                    approx = np.interp(t[mask], coarse["time"], coarse["fluid"][:, j])
                    delta = norm(approx - true[mask], t[mask])
                    coarse_signal = signal
                record(name, horizon, error, delta, signal, coarse_signal)
        if not signed:
            for j, tau in enumerate(SNAP):
                for name, indices in (
                    ("grout", np.flatnonzero(grout)),
                    ("local_ground", np.flatnonzero(~grout)),
                    ("wake", len(local) + np.arange(len(wake))),
                ):
                    if name == "wake" and (case["pe"] == 0 or tau < 10):
                        continue
                    true = fine["field"][j, indices]
                    norm = lambda values: np.sqrt(np.mean(values**2))
                    signal = norm(true)
                    record(
                        name,
                        tau,
                        norm(op["field"][j, indices] - true),
                        norm(coarse["field"][j, indices] - true),
                        signal,
                        signal,
                    )
    return rows
