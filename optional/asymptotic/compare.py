"""Rico and Hermanns low-Péclet benchmark. The asymptotic temperatures are supplied."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from studies.inputs import RESULTS

DATA = Path(__file__).resolve().parent / "data"
OUT = RESULTS / "asymptotic"


def present(recompute=False):
    OUT.mkdir(parents=True, exist_ok=True)
    transient = pd.read_csv(DATA / "asymptotic_transient.csv")
    steady = pd.read_csv(DATA / "asymptotic_steady.csv")
    curve = pd.read_csv(DATA / "asymptotic_steady_curve.csv")
    weights = pd.read_csv(DATA / "asymptotic_errors.csv").set_index("pe")
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
