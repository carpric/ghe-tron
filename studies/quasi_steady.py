"""Compare transient and zero-frequency exterior maps with the same transient interior.

Run from the repository root with python -m studies.quasi_steady.
"""
from pathlib import Path

import numpy as np

from model.boundary import BoundaryOperator, Exterior
from model.quadrature import nodes
from studies.inputs import CASES, RESULTS, save_csv
from studies.single import read_rule

LAYOUTS = ("Q221", "Q422")
FLOWS = (0.5, 1.0, 3.0, 9.0)
OBSERVABLES = ("supply", "return", "wall")
TIMES = np.geomspace(0.02, 100.0, 121)


class PrescribedMap:
    def response(self, s, need_wall):
        return self.matrix, None


def histories(pe, times, refined):
    pipe, wall, radial, padding = (16, 26, 44, 18) if refined else (12, 22, 28, 14)
    models = [BoundaryOperator(dict(CASES[name]["case"], Pe=pe), pipe, wall, radial, padding)
              for name in LAYOUTS]
    exterior = models[0].exterior
    steady = Exterior(pe, models[0].m, radial + 24, padding)
    dc = steady.response(0.0j, False)[0]
    small_s = steady.response(1.e-10 + 0j, False)[0]
    finer = Exterior(pe, models[0].m, radial + 40, padding).response(0.0j, False)[0]
    checks = [np.linalg.norm(dc - other) / np.linalg.norm(dc) for other in (small_s, finer)]
    if max(checks) > 1.e-7:
        raise ArithmeticError(f"Unresolved steady wall map for Pe={pe}: {checks}")
    prescribed = PrescribedMap()
    for model in models:
        model.exterior = prescribed
    if refined:
        z, w = nodes([1.0], 18., 22, 22)
        poles, weights = z[0], w[0] / z[0]
    else:
        poles, weights = read_rule()
    rows = []
    for time in times:
        full_and_qs = np.zeros((2, len(models), 3))
        for pole, weight in zip(poles, weights, strict=True):
            s = pole / time
            for j, matrix in enumerate((exterior.response(s, False)[0], dc)):
                prescribed.matrix = matrix
                for i, model in enumerate(models):
                    common, state = model.frequency(s, True)
                    wall_mean = state[3][wall]
                    full_and_qs[j, i] += (weight * np.array([common, common, wall_mean])).real
        for i, (name, model) in enumerate(zip(LAYOUTS, models, strict=True)):
            full_and_qs[:, i] += [model.right, model.right - 1, 0]
            for h, obs in enumerate(OBSERVABLES):
                full, qs = full_and_qs[:, i, h]
                rows.append(dict(layout=name, pe=pe, tau=time, observable=obs, full=full,
                                 quasi_steady=qs, relative_pct=100*abs(full-qs)/abs(full)))
    if not all(np.isfinite(row["relative_pct"]) for row in rows):
        raise ArithmeticError("Unresolved relative temperature discrepancy")
    return rows, dict(pe=pe, small_s_relative=checks[0], radial_relative=checks[1])


def thresholds(rows):
    results = []
    for name in LAYOUTS:
        for pe in FLOWS:
            for obs in OBSERVABLES:
                curve = [r for r in rows if (r["layout"], r["pe"], r["observable"]) == (name, pe, obs)]
                for tolerance in (1., 5.):
                    outside = [i for i, r in enumerate(curve) if r["relative_pct"] > tolerance]
                    lo = outside[-1] if outside else -1
                    results.append(dict(layout=name, pe=pe, observable=obs, tolerance_pct=tolerance,
                                        last_above=curve[lo]["tau"] if lo >= 0 else "",
                                        first_sustained=curve[lo+1]["tau"] if lo+1 < len(curve) else ""))
    return results


def plot(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator

    style = {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
             "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    colours = ("#163f64", "#007b90", "#b25825", "#704f86")
    labels = ("Supply", "Return", "Mean wall")
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.7), sharex=True, sharey=True)
        for j, name in enumerate(LAYOUTS):
            for i, obs in enumerate(OBSERVABLES):
                ax = axes[i, j]
                ax.set_title(f"({chr(97+2*i+j)}) {labels[i]}, {name}", loc="left", fontsize=10)
                for pe, colour in zip(FLOWS, colours, strict=True):
                    data = [r for r in rows if (r["layout"],r["pe"],r["observable"]) == (name,pe,obs)]
                    ax.loglog([r["tau"] for r in data], [r["relative_pct"] for r in data],
                              color=colour, lw=1.65, label=rf"$\mathrm{{Pe}}={pe:g}$")
                for tol in (1., 5.):
                    ax.axhline(tol, ls="--", c="0.45", lw=.65, zorder=0)
                ax.set_xlim(.02,100)
                ax.set_ylim(.01,200)
                ax.yaxis.set_major_locator(LogLocator(10, numticks=6))
                if j == 0:
                    ax.set_ylabel("Discrepancy (%)")
                if i == 2:
                    ax.set_xlabel(r"Dimensionless time, $\tau$")
        fig.legend(*axes[0,0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False)
        fig.subplots_adjust(left=.13, right=.98, top=.9, bottom=.09, hspace=.37, wspace=.15)
        fig.savefig(destination, bbox_inches="tight", metadata={"CreationDate": None})
        plt.close(fig)


def run(out=RESULTS / "quasi_steady"):
    out = Path(out)
    primary, check = [], []
    for pe in FLOWS:
        rows, diagnostic = histories(pe, TIMES, False)
        primary.extend(rows)
        check.append(diagnostic)
        print(f"Completed Pe={pe}", flush=True)
    save_csv(out / "histories.csv", primary)
    save_csv(out / "thresholds.csv", thresholds(primary))
    save_csv(out / "map_checks.csv", check)
    refinement = []
    for pe in FLOWS:
        rows, _ = histories(pe, TIMES[::10], True)
        refinement.extend(rows)
    save_csv(out / "refined.csv", refinement)
    plot(primary, out / "quasi-steady.pdf")


if __name__ == "__main__":
    run()
