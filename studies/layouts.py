"""Draw exchanger geometry in physical Cartesian axes with analytic potential-flow streamlines."""

import json

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from infer import single_layout
from studies.inputs import CASES, DATA


def cross_section(ax, layout):
    angle = np.deg2rad(layout["flow_angle_deg"])
    if layout["pe"] > 0:
        axis = np.linspace(-2.1, 2.1, 101)
        x, y = np.meshgrid(axis, axis)
        r2 = x * x + y * y
        normal = -x * np.sin(angle) + y * np.cos(angle)
        stream = np.ma.masked_where(r2 <= 1, normal * (1 - 1 / np.maximum(r2, 1)))
        ax.contour(
            x, y, stream, levels=[-1.2, -0.75, -0.35, 0.35, 0.75, 1.2], colors="#85adbd", linewidths=0.6
        )
        ax.annotate(
            "",
            xy=(1.72 * np.cos(angle), 1.72 * np.sin(angle)),
            xytext=(1.1 * np.cos(angle), 1.1 * np.sin(angle)),
            arrowprops=dict(arrowstyle="-|>", color="#166781", lw=1.3),
        )
    ax.add_patch(Circle((0, 0), 1, ec="#565450", fc="#eee9d9", lw=1, zorder=2))
    for b, pair in enumerate(layout["pairs"]):
        for leg, index in enumerate(pair):
            centre = layout["centres"][index]
            ax.add_patch(
                Circle(
                    centre,
                    layout["radii"][index],
                    ec="#324955",
                    fc="#215a72" if leg == 0 else "white",
                    lw=0.9,
                    zorder=3,
                )
            )
            if len(layout["pairs"]) > 1:
                ax.text(
                    *centre,
                    str(b + 1),
                    ha="center",
                    va="center",
                    fontsize=6,
                    zorder=4,
                    color="white" if leg == 0 else "#213746",
                )
    ax.set(xlim=(-1.85, 1.85), ylim=(-1.65, 1.65), aspect="equal")
    ax.set_axis_off()


def panels(folder):
    tables = [
        [dict(single_layout(r["case"]), id=n) for n, r in CASES.items()],
        json.loads((DATA / "parallel_u.json").read_text()),
    ]
    for layouts, cols, filename in zip(tables, (6, 3), ("single-u-layouts", "parallel-u-layouts")):
        fig, axes = plt.subplots(
            len(layouts) // cols,
            cols,
            figsize=(1.65 * cols, 1.55 * len(layouts) // cols + 1),
            layout="constrained",
        )
        for ax, case in zip(axes.flat, layouts, strict=True):
            cross_section(ax, case)
            ax.set_title(f'{case["id"]}    Pe = {case["pe"]:.2g}', fontsize=8)
        fig.suptitle(
            "Supply legs filled, return legs open; arrows indicate groundwater direction", fontsize=10
        )
        fig.savefig(folder / (filename + ".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
        plt.close(fig)
