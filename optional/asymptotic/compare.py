"""Recalculate the low-Péclet asymptotic expansion and compare it with GHE-TRON and FE."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from studies.inputs import RESULTS
from .expansion import inner_problem

DATA = Path(__file__).resolve().parent / "data"
OUT = RESULTS / "asymptotic"


def checked(label, computed, archived):
    computed = np.asarray(computed)
    np.testing.assert_allclose(computed, archived, rtol=0, atol=1e-11, equal_nan=False, err_msg=label)
    difference = np.max(np.abs(computed - np.asarray(archived)))
    print(f"{label}: {computed.size} values, maximum absolute difference {difference:.3g}")
    return computed


def regenerate_expansion(transient, steady, curve, weights):
    """Evaluate at the archived abscissae, then check the resulting ordinates."""
    model = inner_problem(rotation=90.0)
    pairs = [model.weights(pe) for pe in weights.index]
    weights[["v1", "v2"]] = checked("Pipe weights", pairs, weights[["v1", "v2"]])
    for pe, rows in transient.groupby("pe"):
        temperatures = model.temperatures(pe, rows.tau.to_numpy())
        mean = temperatures @ weights.loc[pe, ["v1", "v2"]].to_numpy()
        transient.loc[rows.index, "asymptotic"] = checked(f"Transient Pe={pe:g}", mean, rows.asymptotic)

    pipe_columns = ["pipe1", "pipe2"]
    pairs = [model.temperatures(pe, np.inf) for pe in curve.pe]
    curve[pipe_columns] = checked("Steady curves", pairs, curve[pipe_columns])
    for pipe, rows in steady.groupby("pipe"):
        values = [model.temperatures(pe, np.inf)[pipe - 1] for pe in rows.pe]
        steady.loc[rows.index, "asymptotic"] = checked(f"Steady FE comparison, pipe {pipe}", values, rows.asymptotic)


def present(recompute=False):
    OUT.mkdir(parents=True, exist_ok=True)
    transient = pd.read_csv(DATA / "asymptotic_transient.csv")
    steady = pd.read_csv(DATA / "asymptotic_steady.csv")
    curve = pd.read_csv(DATA / "asymptotic_steady_curve.csv")
    weights = pd.read_csv(DATA / "asymptotic_errors.csv").set_index("pe")
    regenerate_expansion(transient, steady, curve, weights)
    if recompute:
        from .reference import solve_level
        from .operator import exchanger, converged_temperatures

        for kind in ("steady", "transient"):
            solve_level(3, kind)
        fe = pd.read_csv(OUT / "fea_transient_L3.csv")
        for pe, rows in transient.groupby("pe"):
            selected = fe[(fe.pe == pe) & (fe.dt == 0.25)]
            selected = selected[selected.radius == selected.radius.max()].set_index("tau").loc[rows.tau]
            w = weights.loc[pe, ["v1", "v2"]].to_numpy()
            transient.loc[rows.index, "fea"] = selected[["pipe1", "pipe2"]].to_numpy() @ w
            transient.loc[rows.index, "ghe_tron"] = exchanger(pe, 90, OUT / "wall_tables").fluid(rows.tau) @ w
        fe = pd.read_csv(OUT / "fea_steady_L3.csv")
        for pe, rows in steady.groupby("pe"):
            selected = fe[fe.pe == pe].sort_values("radius").iloc[-1]
            idx, values, _, _ = converged_temperatures(pe)
            for i in rows.index:
                pipe = int(steady.loc[i, "pipe"])
                steady.loc[i, ["fea", "ghe_tron"]] = [selected[f"pipe{pipe}"], values[idx, pipe - 1]]
    for method in ("ghe_tron", "asymptotic"):
        steady[method + "_signed_pct"] = 100 * (steady[method] / steady.fea - 1)
    transient.to_csv(OUT / "transient.csv", index=False)
    steady.to_csv(OUT / "steady.csv", index=False)
    curve.to_csv(OUT / "steady_curve.csv", index=False)
    weights[["v1", "v2"]].to_csv(OUT / "weights.csv")
    axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")[1]
    scores = []
    for (pe, rows), color in zip(transient.groupby("pe"), ("#165a77", "#ab492b"), strict=True):
        axes[0].plot(rows.tau, rows.ghe_tron, color=color, label=f"Pe = {pe:g}")
        axes[0].plot(rows.tau, rows.asymptotic, "--", color=color)
        axes[0].plot(rows.tau[::8], rows.fea[::8], "o", mfc="white", color=color, ms=3)

        def error(values):
            return 100 * np.sqrt(
                np.trapezoid((values - rows.fea) ** 2, rows.tau) / np.trapezoid(rows.fea**2, rows.tau)
            )

        scores.append(dict(pe=pe, ghe_tron_pct=error(rows.ghe_tron), asymptotic_pct=error(rows.asymptotic)))
    for (pipe, rows), color in zip(steady.groupby("pipe"), ("#165a77", "#ab492b"), strict=True):
        axes[1].semilogx(rows.pe, rows.ghe_tron, color=color, label=f"Pipe {pipe}")
        axes[1].plot(curve.pe, curve[f"pipe{pipe}"], "--", color=color)
        axes[1].plot(rows.pe, rows.fea, "o", mfc="white", color=color, ms=3)
    axes[0].set(xlabel="Dimensionless time", ylabel="Weighted fluid temperature rise")
    axes[1].set(xlabel="Péclet number", ylabel="Steady pipe temperature rise")
    for ax in axes:
        ax.legend(frameon=False)
    axes[0].figure.suptitle(
        "GHE-TRON (solid), asymptotic expansion (dashed), independent FE (circles)", fontsize=10
    )
    axes[0].figure.savefig(OUT / "asymptotic-comparison.pdf")
    plt.close(axes[0].figure)
    frame = pd.DataFrame(scores)
    frame.to_csv(OUT / "errors.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute", action="store_true", help="Resolve GHE-TRON and the level-three FE references"
    )
    present(parser.parse_args().recompute)
