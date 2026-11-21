"""Numerical figures and error distributions using the published probes and time grids."""

import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from model.loads import reconstruct
from studies.inputs import CASES, PLAN, OPERATING as OPS, DATA, RESULTS
from studies import verification_plots, layouts, circuits

FIG = RESULTS / "figures"
FLOWS = ["Conduction", "Weak flow", "Intermediate flow", "Strong flow"]
BLUE, RUST, INK = "#006596", "#AD492C", "#253A47"


def finish(fig, name):
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight", pad_inches=0.045, metadata={"CreationDate": None})
    plt.close(fig)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.45, color="#dce3e5")
    ax.set_axisbelow(True)


def fluid(get):
    fig, axes = plt.subplots(4, 2, figsize=(6.6, 6.7), layout="constrained", sharex="col")
    for index, name in enumerate(OPS):
        a = get("trained", name)
        b = get("unit_fine", name)
        t = a["tau"]
        np.testing.assert_array_equal(t, b["tau"])
        early = np.flatnonzero(t <= 0.02)
        targets = np.geomspace(t[1], t[-1], 17)
        sampled = 1 + np.abs(np.log(t[1:, None] / targets)).argmin(axis=0)
        for col, (ix, marks) in enumerate([(early, early[::4]), (slice(1, None), sampled)]):
            ax = axes[index, col]
            for j, (color, marker) in enumerate([(BLUE, "o"), (RUST, "s")]):
                ax.plot(t[ix], a["fluid"][ix, j], color=color)
                ax.plot(
                    t[marks],
                    b["fluid"][marks, j],
                    ls="none",
                    marker=marker,
                    ms=3.4,
                    mfc="white",
                    mec=color,
                    mew=0.8,
                )
            ax.yaxis.set_major_locator(MaxNLocator(4))
            style(ax)
        c = CASES[name]["case"]
        axes[index, 0].set_title(FLOWS[index] + r"   " + name, loc="left", fontsize=9)
        axes[index, 1].set_title(
            rf"$\lambda={c['lam']:.2f},\quad \mathrm{{Pe}}={c['Pe']:.3g}$", loc="left", fontsize=9
        )
        axes[index, 0].set(xticks=[0, 0.005, 0.01, 0.015, 0.02], xlim=(-0.0005, 0.0205), ylabel=r"$G_f$")
        axes[index, 1].set(
            xscale="log", xlim=(t[1] * 0.8, t[-1] * 1.2), xticks=[0.001, 0.01, 0.1, 1, 10, 100]
        )
    axes[-1, 0].set_xlabel(r"Startup time $\tau$")
    axes[-1, 1].set_xlabel(r"Full history, $\tau$")
    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, label="GHE-TRON inlet"),
            Line2D([], [], color=RUST, label="GHE-TRON outlet"),
            Line2D([], [], color=BLUE, marker="o", ms=4, mfc="white", ls="", label="FE inlet"),
            Line2D([], [], color=RUST, marker="s", ms=4, mfc="white", ls="", label="FE outlet"),
        ],
        loc="outside upper center",
        ncols=4,
        frameon=False,
    )
    finish(fig, "fluids")


def field_figure(get, wake):
    names = ["Q222", "Q422"]
    fig = plt.figure(figsize=(7.0, 6.7 if wake else 5.45), layout="constrained")
    outer = fig.add_gridspec(2 if wake else 1, 1, height_ratios=[2.25, 1] if wake else [1])
    axes = outer[0].subgridspec(2, 3).subplots()
    if wake:
        profiles = outer[1].subgridspec(1, 2).subplots()
    for i, name in enumerate(names):
        p = CASES[name]["case"]
        a, b = [get(key, name) for key in ("trained", "unit_fine")]
        if wake:
            x = np.linspace(3, 25, 89)
            y = np.linspace(-8, 8, 65)
            model, ref = [d["wake10"].reshape(len(y), len(x)) for d in (a, b)]
        else:
            x = y = np.linspace(-3, 3, 121)
            X, Y = np.meshgrid(x, y)
            keep = np.ones(X.shape, dtype=bool)
            for j in [1, 2]:
                keep &= np.hypot(X - p[f"xp{j}"], Y - p[f"yp{j}"]) > p[f"Rp{j}"] + 1e-4
            model, ref = np.full((2, *X.shape), np.nan)
            model[keep] = a["ground10"]
            ref[keep] = b["ground10"]
        lim = max(float(np.nanmax(np.abs(model - ref))), 1e-14)
        norm = Normalize(min(0.0, np.nanmin([model, ref])), np.nanmax([model, ref]))
        for c, (ax, fld) in enumerate(zip(axes[i], [ref, model, model - ref])):
            vmax = lim if c == 2 else norm.vmax
            scale = TwoSlopeNorm(0, -lim, lim) if c == 2 else norm
            im = ax.pcolormesh(
                x,
                y,
                fld,
                shading="nearest",
                cmap="RdBu_r" if c == 2 else "inferno",
                norm=scale,
                rasterized=True,
            )
            ax.set_aspect("equal")
            ax.set_title(["Fine FE", "GHE-TRON", "GHE-TRON minus FE"][c], fontsize=9)
            if c < 2:
                levels = [level for level in [0.05, 0.2] if np.nanmin(fld) < level < np.nanmax(fld)]
                if levels:
                    ax.contour(x, y, fld, levels=levels, colors="white", linewidths=0.50, alpha=0.85)
            if wake:
                ax.set_xticks([3, 10, 17, 25])
                ax.set_yticks([-8, 0, 8])
            else:
                ax.add_patch(Circle((0, 0), 1, ec="white", fc="none", lw=0.5))
                for j in [1, 2]:
                    ax.add_patch(Circle((p[f"xp{j}"], p[f"yp{j}"]), p[f"Rp{j}"], fc="white", ec=INK, lw=0.5))
                ax.set_xticks([-3, 0, 3])
                ax.set_yticks([-3, 0, 3])
            if i == 1:
                ax.set_xlabel(r"$x_\parallel^d/R_0$" if wake else r"$x^d/R_0$")
            if c == 0:
                ax.set_ylabel(name + "\n" + (r"$y_\perp^d/R_0$" if wake else r"$y^d/R_0$"))
            if c:
                ticks = [-vmax, 0, vmax] if c == 2 else [0, 0.5 * vmax, vmax]
                bar = fig.colorbar(
                    im,
                    ax=list(axes[i, :2]) if c == 1 else ax,
                    location="bottom",
                    shrink=0.82,
                    aspect=38,
                    pad=0.05,
                    ticks=ticks,
                )
                bar.ax.set_xticklabels([f"{v:.2g}" for v in ticks])
                bar.ax.tick_params(labelsize=7, pad=1)
                bar.set_label(r"$\Delta\theta$" if c == 2 else r"$\theta$", fontsize=8, labelpad=0)
        if wake:
            color, mark = [(BLUE, "o"), (RUST, "s")][i]
            for ax, coord, cut in [(profiles[0], x, (32, slice(None))), (profiles[1], y, (slice(None), 28))]:
                ax.plot(coord, model[cut], color=color, lw=1.45, label=name)
                ax.plot(
                    coord[::4],
                    ref[cut][::4],
                    ls="none",
                    mfc="white",
                    mec=color,
                    marker=mark,
                    ms=3.2,
                    mew=0.75,
                )
                ax.set_ylabel(r"$G_s$")
                style(ax)
    if wake:
        profiles[0].set(xlabel=r"$x_\parallel$ at $y_\perp=0$", xlim=(3, 25))
        profiles[1].set(xlabel=r"$y_\perp$ at $x_\parallel=10$", xlim=(-8, 8))
        profiles[0].set_title("Centreline", loc="left")
        profiles[1].set_title("Transverse profile", loc="left")
        profiles[0].legend(frameon=False, fontsize=8)
    finish(fig, "wake" if wake else "local")


def operating(get, signed_scores):
    fig, axes = plt.subplot_mosaic(
        [["q", "q"], ["a", "b"], ["c", "d"]],
        figsize=(7.2, 5.0),
        height_ratios=[0.62, 1, 1],
        layout="constrained",
    )
    x = np.r_[0, PLAN["switches"], 100]
    y = np.r_[PLAN["levels"], PLAN["levels"][-1]]
    axes["q"].step(x, y, where="post", color=INK, lw=1.4)
    axes["q"].axhline(0, color="#777777", lw=0.5)
    axes["q"].set_ylabel(r"Load $\hat q$")
    axes["q"].set_xlim(0, 100)
    style(axes["q"])
    for i, name in enumerate(OPS):
        ax = axes["abcd"[i]]
        a = get("trained", name)
        b = get("signed_fine", name)
        t = b["tau"]
        q = np.take(PLAN["levels"], np.searchsorted(PLAN["switches"], t - 1e-10))
        q[t == 0] = 0
        kernel = CubicSpline(a["tau"], a["fluid"], axis=0)(t)
        response = reconstruct(kernel, q)
        for j, col in enumerate([BLUE, RUST]):
            ax.plot(t, response[:, j], color=col, lw=1.1)
            sample = np.arange(13, len(t), 79)
            ax.plot(t[sample], b["fluid"][sample, j], ["o", "s"][j], ms=2.8, color=col, mfc="white", mew=0.6)
            expected = next(
                float(r["rms"])
                for r in signed_scores
                if r["case"] == name and r["observable"] == ["signed inlet", "signed outlet"][j]
            )
            np.testing.assert_allclose(
                np.sqrt(np.trapezoid((response[:, j] - b["fluid"][:, j]) ** 2, t) / 100),
                expected,
                rtol=1e-8,
                atol=1e-11,
            )
        ax.set_title(name + "   " + FLOWS[i], loc="left")
        ax.set_xlim(0, 100)
        style(ax)
        if i % 2 == 0:
            ax.set_ylabel(r"$\Theta_f$")
        if i >= 2:
            ax.set_xlabel(r"Dimensionless time $\tau$")
    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, label="Inlet"),
            Line2D([], [], color=RUST, label="Outlet"),
            Line2D([], [], color=BLUE, marker="o", mfc="white", ls="none", ms=4, label="FE inlet"),
            Line2D([], [], color=RUST, marker="s", mfc="white", ls="none", ms=4, label="FE outlet"),
        ],
        loc="outside upper center",
        ncol=4,
        frameon=False,
    )
    finish(fig, "operating")


def groundwater_sensitivity(data):
    with (data / "groundwater_fluids.csv").open() as file:
        fluids = list(csv.DictReader(file))
    with (data / "groundwater_probes.csv").open() as file:
        ground = list(csv.DictReader(file))
    levels = sorted({float(r["Pe"]) for r in fluids})
    fig = plt.figure(figsize=(6.65, 5.35), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    ax = fig.add_subplot(grid[0, :])
    colors = [INK, "#96733c", BLUE, RUST]
    dashes = ["-", (0, (5, 2)), (0, (3, 1, 1, 1)), "-"]
    for i, (pe, color, linestyle) in enumerate(zip(levels, colors, dashes, strict=True)):
        for source in ("GHE-TRON", "fine FE"):
            rr = [r for r in fluids if float(r["Pe"]) == pe and r["source"] == source]
            hours = np.array([24 * float(r["time_days"]) for r in rr])
            inlet = np.array([float(r["inlet_C"]) for r in rr])
            if source == "GHE-TRON":
                ax.plot(
                    hours[1:],
                    inlet[1:],
                    color=color,
                    ls=linestyle,
                    lw=1.65,
                    label=rf"$\mathrm{{Pe}}={pe:.3g}$",
                )
            else:
                # Offset the marker times so the two weakly separated curves remain legible.
                targets = np.geomspace(0.006, 150.0, 13)[1:-1] * 1.16**i
                idx = np.unique(1 + abs(np.log(hours[1:, None] / targets)).argmin(axis=0))
                ax.plot(hours[idx], inlet[idx], "o", color=color, ms=3.2, mfc="white", mew=0.8)
    ax.plot([], [], "o", color=INK, mfc="white", ms=3.5, label="Fine FE")
    ax.set(xscale="log", xlim=(0.00065, 175), ylabel=r"Inlet temperature ($^\circ$C)")
    ax.set_xticks([1 / 60, 1, 24, 156.25], ["1 min", "1 h", "1 day", "6.51 days"])
    ax.set_xlabel("Elapsed time after the start of injection")
    ax.set_title("(a) Fluid response under a sustained 5 kW injection", loc="left")
    ax.legend(ncols=3, loc="upper left", frameon=False, fontsize=8, columnspacing=0.9)
    style(ax)
    ymax = max(float(r[k]) for r in ground for k in ("upstream_rise_K", "downstream_rise_K"))
    other = None
    for col, (tau, title) in enumerate(
        [(10, "(b) Ground after 15.6 h"), (100, "(c) Ground after 6.51 days")]
    ):
        ax = fig.add_subplot(grid[1, col], sharey=other)
        other = ax
        x = np.arange(len(levels))
        for key, dx, marker, color, label in [
            ("upstream_rise_K", -0.105, "s", RUST, "Upstream"),
            ("downstream_rise_K", 0.105, "o", BLUE, "Downstream"),
        ]:
            for source in ("GHE-TRON", "fine FE"):
                values = {
                    float(r["Pe"]): float(r[key])
                    for r in ground
                    if r["source"] == source and float(r["tau"]) == tau
                }
                pred = source == "GHE-TRON"
                ax.scatter(
                    x + dx,
                    [values[p] for p in levels],
                    marker=marker,
                    s=19 if pred else 56,
                    facecolors=color if pred else "none",
                    edgecolors=color,
                    linewidths=0.8,
                    label=label if pred else None,
                )
        ax.set(
            xticks=x,
            xticklabels=[f"{p:.3g}" for p in levels],
            xlim=(-0.45, 3.45),
            xlabel=r"Péclet number, $\mathrm{Pe}$",
        )
        ax.set_ylim(-0.035 * ymax, 1.12 * ymax)
        ax.set_title(title, loc="left")
        if col == 0:
            ax.set_ylabel("Ground temperature rise (K)")
            ax.legend(loc="upper left", fontsize=8, frameon=False)
        else:
            ax.tick_params(labelleft=False)
        style(ax)
    finish(fig, "groundwater-sensitivity")


def figures(data):
    FIG.mkdir(exist_ok=True, parents=True)
    computed = data == RESULTS

    def get(method, layout):
        if computed:
            key = {"trained": "operator", "unit_fine": "fine", "signed_fine": "signed_fine"}[method]
            with np.load(RESULTS / "unit" / layout / (key + ".npz")) as z:
                arrays = {k: z[k] for k in ("fluid", "tau")}
                if layout in ("Q222", "Q422") and key != "signed_fine":
                    arrays.update(ground10=z["local"][3], wake10=z["wake"][0])
                return arrays
        prefix = f"{layout}_{method}_"
        with np.load(DATA / "figure_samples.npz") as z:
            return {k[len(prefix) :]: z[k] for k in z.files if k.startswith(prefix)}

    fluid(get)
    field_figure(get, False)
    field_figure(get, True)
    operating(get, pd.read_csv(data / "operating_fine_grid.csv").to_dict("records"))
    groundwater_sensitivity(data)

    def verification_file(reported, calculated):
        return RESULTS / "verification" / calculated if computed else data / reported

    verification_plots.inversion(
        verification_file("verification_inverse.csv", "inversion.csv"), FIG / "inversion.pdf"
    )
    verification_plots.timing(
        verification_file("verification_refinement.csv", "refinement.csv"),
        verification_file("verification_timing.csv", "timing.csv"),
        FIG / "error-work.pdf",
    )

    df = pd.read_csv(data / "square_return_histories.csv")
    fig, axes = plt.subplots(3, 2, figsize=(8, 8), layout="constrained")
    for ax, (kind, pairs) in zip(axes[0], circuits.PAIRS.items(), strict=True):
        layouts.cross_section(
            ax,
            dict(
                centres=circuits.CENTRES,
                radii=circuits.RADII,
                pairs=pairs,
                pe=2.0,
                flow_angle_deg=np.degrees(circuits.ANGLE),
            ),
        )
        ax.set_title(f"{kind.capitalize()} supply legs")
    measurements = pd.read_csv(data / "square_observations.csv")
    for ax, quantity, label in zip(
        axes[2],
        ("branch_spread", "mixed_return"),
        ("Branch temperature difference", "Mixed return temperature"),
    ):
        for kind, color in (("adjacent", BLUE), ("opposite", RUST)):
            for source, marker in (("GHE-TRON", "o"), ("FE fine", "s")):
                y = []
                for pe in (0, 2, 5):
                    rows = measurements[
                        (measurements["case"] == circuits.name(kind, pe, 0.5))
                        & (measurements.source == source)
                        & (measurements.tau == 10.0)
                    ]
                    y.append(rows[quantity].item())
                ax.plot(
                    [0, 2, 5],
                    y,
                    marker=marker,
                    color=color,
                    ls="-" if source == "GHE-TRON" else "none",
                    mfc=color if source == "GHE-TRON" else "none",
                    label=f"{kind}, {source}",
                )
        ax.set(xlabel="Péclet number", ylabel=label, title="Equal flow split at τ = 10", xticks=[0, 2, 5])
        ax.legend(fontsize=7, frameon=False)

    for ax, (share, group) in zip(axes[1], df.groupby("first_branch_fraction"), strict=True):
        for j, color in ((1, BLUE), (2, RUST)):
            ax.plot(group.tau, group[f"return_{j}_operator"], color=color, label=f"Return {j}")
            marks = group.iloc[::40]
            ax.plot(marks.tau, marks[f"return_{j}_fine_FE"], "o", ms=3, mfc="white", mec=color)
        ax.set(
            xlabel="Dimensionless time",
            ylabel="Return temperature rise",
            title=f"Mass-flow split {share:.0%} / {1-share:.0%}",
        )
        ax.legend(frameon=False)
    finish(fig, "double-u")
    layouts.panels(FIG)
    print(FIG)


def summary(data):
    d = pd.read_csv(data / "casewise_FE.csv")
    contrasts = d.pivot(
        index=["layout", "flow", "observable", "time"], columns="comparison", values="relative_pct"
    )
    contrasts["within_FE_pair"] = contrasts.operator_minus_fine_FE <= contrasts.coarse_minus_fine_FE

    def aggregate(frame):
        return frame.groupby(["observable", "time"], sort=False).agg(
            layouts=("operator_minus_fine_FE", "size"),
            median_pct=("operator_minus_fine_FE", "median"),
            p90_pct=("operator_minus_fine_FE", lambda s: s.quantile(0.90)),
            max_pct=("operator_minus_fine_FE", "max"),
            median_FE_difference_pct=("coarse_minus_fine_FE", "median"),
            within_FE_pair=("within_FE_pair", "sum"),
        )

    target = RESULTS / "summary"
    target.mkdir(exist_ok=True, parents=True)
    report = aggregate(contrasts.reset_index())
    report.to_csv(target / "single_u.csv")
    for flow, frame in contrasts.reset_index().groupby("flow"):
        aggregate(frame).to_csv(target / (flow + ".csv"))
    print(report.loc[[(quantity, 100.0) for quantity in ("inlet", "outlet", "wake")]].to_string())
    print(target)
