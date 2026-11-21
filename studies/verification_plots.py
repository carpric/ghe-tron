"""Error versus inversion order and wall-clock cost at matched accuracy."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd

NAVY, RUST, GREEN = "#165a77", "#a4502a", "#327465"


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dce2e4", lw=0.6)


def inversion(source, destination):
    data = pd.read_csv(source)
    fig, axes = plt.subplots(1, 2, figsize=(6.55, 3.05), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.32, top=0.86, wspace=0.13)
    for ax, obs, title in zip(axes, ("fluid", "wake"), ("(a) Fluid temperatures", "(b) Downstream ground")):
        frame = data[data.observable == obs]
        refs = frame.drop_duplicates(["case", "tau"]).reference_difference
        ax.axhline(refs.median(), color="#686868", ls=":", lw=1.3)
        for method, color, marker in (("learned", NAVY, "o"), ("Talbot", RUST, "s")):
            grouped = frame[frame.method == method].groupby("J").rms
            ax.plot(grouped.median(), color=color, marker=marker, ms=3.7, lw=1.35)
            ax.plot(grouped.max(), color=color, ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel(r"Complex solves per time anchor, $J$")
        ax.set_xticks(np.arange(4, 13, 2))
        ax.set_xlim(3.8, 12.2)
        style(ax)
    axes[0].set_ylabel("Absolute RMS discrepancy")
    handles = [
        Line2D([], [], color=NAVY, marker="o", ms=4, label="Fitted rule, median"),
        Line2D([], [], color=RUST, marker="s", ms=4, label="Modified Talbot, median"),
        Line2D([], [], color="0.25", ls="--", label="Maximum in each family"),
        Line2D([], [], color="0.4", ls=":", label="Median 37--45 reference difference"),
    ]
    fig.legend(
        handles=handles,
        ncol=2,
        loc="lower center",
        frameon=False,
        columnspacing=1.8,
        handlelength=2.1,
        bbox_to_anchor=(0.53, 0.005),
    )
    fig.savefig(destination, metadata={"CreationDate": None})
    plt.close(fig)


def timing(refinement, timing, destination):
    data = pd.read_csv(refinement)
    runtime = pd.read_csv(timing).set_index(["case", "method"])
    fig, axes = plt.subplots(1, 2, figsize=(6.55, 3.12), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.32, top=0.86, wspace=0.28)
    cases = ["Q122", "Q222", "Q322", "Q422"]
    colors = [NAVY, GREEN, RUST, "#745483"]
    for ax, obs, time, title in zip(
        axes, ("inlet", "wake"), (100, 10), ("(a) Inlet, full history", r"(b) Downstream ground, $\tau=10$")
    ):
        frame = data[(data.observable == obs) & (data.tau == time)].set_index("case")
        for name, color in zip(cases, colors):
            if name not in frame.index:
                continue
            row = frame.loc[name]
            y = row.model_vs_FE3_pct
            warm = runtime.loc[(name, "operator"), "seconds"]
            first = runtime.loc[(name, "operator"), "first_use_s"]
            ax.plot([warm, first], [y, y], color=color, lw=1.2)
            ax.plot(first, y, marker="o", ms=5, mec=color, mfc="white", ls="none")
            ax.plot(warm, y, marker="o", ms=3.0, color=color, ls="none")
            for level, err, marker in (("1", row.coarse_vs_FE3_pct, "^"), ("2", row.fine_vs_FE3_pct, "s")):
                ax.plot(
                    runtime.loc[(name, level), "seconds"], err, marker=marker, color=color, ms=4, ls="none"
                )
        ax.set(xscale="log", yscale="log", xlabel="Wall time (s)")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, loc="left")
        style(ax)
    axes[0].set_ylabel(r"Relative RMS against FE level 3 (%)")
    handles = [Line2D([], [], color=c, lw=2, label=n) for n, c in zip(cases, colors)]
    handles += [
        Line2D([], [], marker=m, ls="none", color="0.18", markerfacecolor=face, ms=4.5, label=label)
        for m, face, label in (
            ("o", "0.18", "Operator, reused"),
            ("o", "white", "Operator, first use"),
            ("^", "0.18", "FE coarse"),
            ("s", "0.18", "FE fine"),
        )
    ]
    fig.legend(
        handles=handles,
        ncol=4,
        loc="lower center",
        frameon=False,
        columnspacing=1.1,
        handlelength=1.1,
        bbox_to_anchor=(0.53, 0.005),
    )
    fig.savefig(destination, metadata={"CreationDate": None})
    plt.close(fig)
