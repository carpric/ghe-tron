"""Unit-step and signed-load assessment on the 24 registered single-U layouts."""

import numpy as np
from scipy.interpolate import CubicSpline
from model.loads import reconstruct
from model.wall import WallTable
from model.quadrature import nodes
from studies.inputs import CASES, OPERATING, RESULTS, GRID, command, probes, time_nodes
from studies.single import predict, read_rule, rows_for

RAW = RESULTS / "unit"
TABLES = RESULTS / "wall_tables"


def calculate_reference(case, level, signed=False, *, radius=50):
    from fea.reference import make_solver, solve, field_samples, fluid

    factor = 2 ** (level - 1)
    model = make_solver(case, dict(mesh=level, radius=radius))
    times = np.linspace(0, 100, 1000 * factor + 1) if signed else time_nodes(factor)
    snapshots = None if signed else GRID["snapshots"]
    solution, _ = solve(model, times, command if signed else lambda _: 1.0, snapshots)
    np.testing.assert_allclose(solution["times"], times, atol=1e-12, rtol=0)
    history = fluid(solution)
    np.testing.assert_allclose(history[0], 0.0, atol=1e-12)
    arrays = dict(tau=times, fluid=history)
    if signed:
        np.testing.assert_allclose(history[:, 0] - history[:, 1], command(times), atol=1e-9)
    else:
        right = (1 + case["psi2"]) / (case["psi1"] + case["psi2"])
        history[0] = [right, right - 1]
        xy, downstream = probes(case)
        arrays["local"] = field_samples(model, solution, snapshots, xy)
        arrays["wake"] = (
            field_samples(model, solution, GRID["wake_times"], downstream) if case["Pe"] else np.empty((3, 0))
        )
    return arrays, model.V.dofmap.index_map.size_global


def store(name, kind, calculate):
    path = RAW / name / (kind + ".npz")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = calculate()
        if not all(np.isfinite(x).all() for x in arrays.values()):
            raise ArithmeticError(f"Nonfinite result for {name}, {kind}")
        temp = path.with_suffix(".part.npz")
        np.savez_compressed(temp, **arrays)
        temp.replace(path)
    with np.load(path) as archive:
        return dict(archive)


def signed_prediction(data, times):
    kernel = CubicSpline(data["tau"], data["fluid"], axis=0)(times)
    return dict(tau=times, fluid=reconstruct(kernel, command(times)))


def signed_metrics(reference, candidate):
    times = reference["tau"]
    np.testing.assert_allclose(times, candidate["tau"], atol=1e-12)

    def norm(v):
        return np.sqrt(np.trapezoid(v * v, times) / times[-1])

    for col, label in enumerate(("signed supply", "signed return")):
        truth = reference["fluid"][:, col]
        yield label, times[-1], norm(candidate["fluid"][:, col] - truth), norm(truth)


def unit_step_responses(name, table):
    case = CASES[name]["case"]
    operator = store(name, "operator", lambda: predict(case, read_rule(), table))
    coarse = store(name, "coarse", lambda: calculate_reference(case, 1)[0])
    fine = store(name, "fine", lambda: calculate_reference(case, 2)[0])
    return operator, coarse, fine


def unit(names):
    records, operating = [], []
    table = WallTable(TABLES, 22, 28, 14)
    for name in names:
        case = CASES[name]["case"]
        result, coarse, fine = unit_step_responses(name, table)
        if name in OPERATING:
            rough = store(name, "signed_coarse", lambda: calculate_reference(case, 1, True)[0])
            truth = store(name, "signed_fine", lambda: calculate_reference(case, 2, True)[0])
            signed = signed_prediction(result, truth["tau"])
            for obs, time, err, scale in signed_metrics(truth, signed):
                operating.append(
                    dict(
                        case=name,
                        observable=obs,
                        rms=err,
                        reference_rms=scale,
                        relative_pct=100 * err / scale,
                    )
                )
            common = {k: v[::2] for k, v in truth.items()}
        for label, values in (("coarse_minus_fine_FE", coarse), ("operator_minus_fine_FE", result)):
            metrics = list(rows_for(fine, values, case))
            if name in OPERATING:
                compare = rough if values is coarse else {k: v[::2] for k, v in signed.items()}
                metrics.extend(signed_metrics(common, compare))
            for obs, time, err, scale in metrics:
                records.append(
                    dict(
                        layout=name,
                        flow=CASES[name]["flow"],
                        observable=obs,
                        time=time,
                        comparison=label,
                        rms=err,
                        fine_FE_rms=scale,
                        relative_pct=100 * err / scale,
                    )
                )
        print(f"Assessed {name}", flush=True)
    return records, operating


def temporal(names):
    """Compare fixed and fitted poles against 37- and 45-frequency Euler inversions."""
    table = WallTable(TABLES, 22, 28, 14)
    records = []
    for name in names:
        case = CASES[name]["case"]
        evaluations = {}
        for count, A in ((37, 16.0), (45, 18.0)):
            s, w = nodes([1.0], A, (count - 1) // 2, (count - 1) // 2)
            evaluations[count] = predict(case, (s[0], w[0] / s[0]), table)
        for label, rule in (("fixed", "fixed.npz"), ("optimized", "fitted.npz")):
            evaluations[label] = predict(case, read_rule(rule), table)
        base = evaluations[45]
        by = {}
        for key in (37, "fixed", "optimized"):
            values = evaluations[key]
            metrics = list(rows_for(base, values, case))
            if name in OPERATING:
                t = np.linspace(0, 100, 2001)
                metrics.extend(signed_metrics(signed_prediction(base, t), signed_prediction(values, t)))
            by[key] = {(obs, t): (err, signal) for obs, t, err, signal in metrics}
        for (obs, t), (diff, scale) in by[37].items():
            fixed = by["fixed"][obs, t][0]
            optimized = by["optimized"][obs, t][0]
            tol = max(diff, 1e-11)
            records.append(
                dict(
                    case=name,
                    flow=CASES[name]["flow"],
                    observable=obs,
                    time=t,
                    fixed_rms=fixed,
                    optimized_rms=optimized,
                    numerical_difference=diff,
                    reference_signal=scale,
                    fixed_pct=100 * fixed / scale,
                    optimized_pct=100 * optimized / scale,
                    resolved_gain=fixed - optimized > tol,
                    resolved_loss=optimized - fixed > tol,
                )
            )
    return records
