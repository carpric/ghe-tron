"""Generate frozen predictions, regression controls, and casewise assessment."""

import csv
import numpy as np
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
from model.response import Circuit, MultiPipeResponse

RAW = HERE / "raw"


def model(kind, pe, fraction, centres=CENTRES):
    return MultiPipeResponse(
        centres,
        RADII,
        BETA,
        COUPLING,
        LAMBDA,
        GAMMA,
        pe,
        ANGLE,
        Circuit.parallel_utubes(PAIRS[kind], [fraction, 1 - fraction]),
        RAW / "tables",
    )


def outputs(tube, pairs, fractions):
    pairs = np.array(pairs)
    inlet = tube[:, pairs[:, 0]]
    returns = tube[:, pairs[:, 1]]
    np.testing.assert_allclose(inlet, inlet[:, 0:1] + np.zeros_like(inlet), atol=2e-11, rtol=0)
    mix = returns @ fractions
    np.testing.assert_allclose(inlet[:, 0] - mix, 1.0, atol=2e-11, rtol=0)
    return np.column_stack((inlet[:, 0], returns, mix))


def predict():
    local, down = probes()
    t = times(2)
    RAW.mkdir(parents=True, exist_ok=True)
    for kind, pe, f in CASES:
        op = model(kind, pe, f)
        fluid = outputs(op.fluid(t), PAIRS[kind], [f, 1 - f])
        port = op.circuit_temperatures(t)
        np.testing.assert_allclose(
            fluid, np.column_stack((port["supply"], port["returns"], port["mixed_return"]))
        )
        xy = np.vstack([local, down])
        field = np.array([op.fields(tau, xy) for tau in SNAP])
        np.savez_compressed(RAW / (name(kind, pe, f) + "_op.npz"), time=t, fluid=fluid, field=field)
        print("Predicted", name(kind, pe, f), flush=True)


def write_csv(filename, records):
    with (HERE / filename).open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def measure():
    local, down = probes()
    is_grout = np.linalg.norm(local, axis=1) < 1.0
    groups = {
        "grout": np.where(is_grout)[0],
        "local_ground": np.where(~is_grout)[0],
        "wake": len(local) + np.arange(len(down)),
    }
    scores = []
    endpoints = []
    columns = ["supply", "return_1", "return_2", "mixed_return"]

    def row(case, quantity, t, err, delta, signal):
        scores.append(
            dict(
                case=case,
                quantity=quantity,
                tau=t,
                absolute_rms=err,
                relative_pct=100 * err / signal,
                coarse_fine_rms=delta,
                coarse_fine_pct=100 * delta / signal,
                reference_rms=signal,
            )
        )

    for kind, pe, f in CASES:
        case = name(kind, pe, f)
        with (
            np.load(RAW / (case + "_fe2.npz")) as fine,
            np.load(RAW / (case + "_fe1.npz")) as coarse,
            np.load(RAW / (case + "_op.npz")) as op,
        ):
            t = fine["time"]
            np.testing.assert_allclose(t, op["time"], atol=1e-12, rtol=0)
            for col, quantity in enumerate(columns):
                true = fine["fluid"][:, col]
                pred = op["fluid"][:, col]
                rough = np.interp(t, coarse["time"], coarse["fluid"][:, col])
                for horizon in (0.02, 2.0, 100.0):
                    mask = t <= horizon

                    def norm(v):
                        return np.sqrt(np.trapezoid(v[mask] ** 2, t[mask]) / horizon)

                    row(case, quantity, horizon, norm(pred - true), norm(rough - true), norm(true))
            for j, tau in enumerate(SNAP):
                for quantity, idx in groups.items():
                    if quantity == "wake" and (pe == 0.0 or tau < 10):
                        continue
                    pred = op["field"][j, idx]
                    true = fine["field"][j, idx]
                    rough = coarse["field"][j, idx]
                    norm = lambda v: np.sqrt(np.mean(v**2))
                    row(case, quantity, tau, norm(pred - true), norm(rough - true), norm(true))
            for source, arr in [("GHE-TRON", op), ("FE fine", fine), ("FE coarse", coarse)]:
                for tau in SNAP:
                    k = np.flatnonzero(np.isclose(arr["time"], tau, atol=1e-12, rtol=0)).item()
                    vals = arr["fluid"][k]
                    j = [*SNAP].index(tau)
                    # First column on wake centreline x_parallel=3, y_perp=0.
                    point = np.argmin(
                        np.sum((down - 3 * np.array([np.cos(ANGLE), np.sin(ANGLE)])) ** 2, axis=1)
                    )
                    endpoints.append(
                        dict(
                            case=case,
                            source=source,
                            tau=tau,
                            **dict(zip(columns, vals)),
                            branch_spread=abs(vals[1] - vals[2]),
                            wake_x3_y0=arr["field"][j, len(local) + point],
                        )
                    )
    write_csv("metrics.csv", scores)
    write_csv("observations.csv", endpoints)
    read = {(r["case"], r["source"], r["tau"]): r for r in endpoints}
    contrasts = []
    comparisons = [(name("adjacent", pe, 0.5), name("opposite", pe, 0.5)) for pe in (0.0, 2.0, 5.0)]
    comparisons.append((name("opposite", 2.0, 0.5), name("opposite", 2.0, 0.35)))
    for a, b in comparisons:
        for t in SNAP:
            for obs in ("mixed_return", "branch_spread", "wake_x3_y0"):
                delta = [
                    read[b, s, t][obs] - read[a, s, t][obs] for s in ("GHE-TRON", "FE fine", "FE coarse")
                ]
                contrasts.append(
                    dict(
                        baseline=a,
                        changed=b,
                        tau=t,
                        observable=obs,
                        predicted_change=delta[0],
                        fine_change=delta[1],
                        coarse_change=delta[2],
                        refinement_of_change=abs(delta[1] - delta[2]),
                    )
                )
    write_csv("matched_contrasts.csv", contrasts)


def signed():
    case = name("opposite", 2.0, 0.35)
    op = model("opposite", 2.0, 0.35)
    with (
        np.load(RAW / ("signed_" + case + "_fe2.npz")) as fine,
        np.load(RAW / ("signed_" + case + "_fe1.npz")) as coarse,
    ):
        t = fine["time"]
        command = signed_command(t)
        port = op.circuit_temperatures(t, command)
        data = np.column_stack((port["supply"], port["returns"], port["mixed_return"]))
        np.testing.assert_allclose(data[:, 0] - data[:, -1], command, atol=2e-11, rtol=0)
        np.testing.assert_allclose(t[::2], coarse["time"], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(signed_command(t)[::2], signed_command(coarse["time"]))
        results = []
        for j, obs in enumerate(("supply", "return_1", "return_2", "mixed_return")):
            truth = fine["fluid"][:, j]
            coarse_error = coarse["fluid"][:, j] - truth[::2]

            def norm(v, t):
                return np.sqrt(np.trapezoid(v**2, t) / t[-1])

            error = norm(data[:, j] - truth, t)
            results.append(
                dict(
                    observable=obs,
                    relative_pct=100 * error / norm(truth, t),
                    absolute_rms=error,
                    coarse_fine_pct_common=100 * norm(coarse_error, t[::2]) / norm(truth[::2], t[::2]),
                )
            )
        write_csv("signed_metrics.csv", results)
