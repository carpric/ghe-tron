"""Predict dimensionless fluid temperatures and solid fields for a parallel-U circuit."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from model.response import Circuit, MultiPipeResponse, RULE
from studies.inputs import CASES, DATA, RESULTS


def single_layout(c):
    return dict(
        centres=[[c[f"xp{k}"], c[f"yp{k}"]] for k in (1, 2)],
        radii=[c["Rp1"], c["Rp2"]],
        beta=[c["beta1"], c["beta2"]],
        coupling=c["psi1"] * c["beta1"] / c["Rp1"],
        lam=c["lam"],
        gamma=c["gamma"],
        pe=c["Pe"],
        flow_angle_deg=np.rad2deg(c["phi_v"]),
        pairs=[[0, 1]],
        fractions=[1.0],
    )


def read_layout(name):
    if Path(name).is_file():
        return json.loads(Path(name).read_text())
    if name in CASES:
        return single_layout(CASES[name]["case"])
    for layout in json.loads((DATA / "parallel_u.json").read_text()):
        if layout["id"] == name:
            return layout
    raise ValueError(f"Unknown layout {name!r}. Use a registered ID or a layout JSON file.")


def bind(c, cache=RESULTS / "wall_tables", rule_file=RULE):
    circuit = Circuit.parallel_utubes(c["pairs"], c["fractions"])
    return MultiPipeResponse(
        c["centres"],
        c["radii"],
        c["beta"],
        c["coupling"],
        c["lam"],
        c["gamma"],
        c["pe"],
        np.deg2rad(c["flow_angle_deg"]),
        circuit,
        cache,
        rule_file,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("layout", nargs="?", default="Q221", help="Layout ID or JSON path")
    p.add_argument("--times", nargs="+", type=float, default=[0, 0.02, 1, 10, 100], help="Unit-step times")
    p.add_argument("--load", type=Path, help="CSV with tau,q on a uniform grid, beginning at 0,0")
    p.add_argument(
        "--field",
        nargs=3,
        type=float,
        metavar=("TAU", "X", "Y"),
        action="append",
        help="Query the unit-step solid field in borehole-radius units; repeat for more points",
    )
    p.add_argument("--rule", type=Path, default=RULE, help="NPZ containing complex nodes z and residues w")
    p.add_argument("--direct", action="store_true", help="Calculate the wall map directly at each frequency")
    p.add_argument("--output", type=Path, default=RESULTS / "prediction")
    args = p.parse_args()
    op = bind(read_layout(args.layout), None if args.direct else RESULTS / "wall_tables", args.rule)
    time, load = args.times, None
    if args.load:
        data = np.genfromtxt(args.load, delimiter=",", names=True, ndmin=1)
        time, load = data["tau"], data["q"]
    fluid = op.circuit_temperatures(time, load)
    args.output.mkdir(parents=True, exist_ok=True)
    columns = ["tau", "supply", *[f"return_{j+1}" for j in range(len(op.circuit.branches))], "mixed_return"]
    np.savetxt(
        args.output / "fluids.csv",
        np.column_stack((time, fluid["supply"], fluid["returns"], fluid["mixed_return"])),
        delimiter=",",
        header=",".join(columns),
        comments="",
    )
    if args.field:
        points = np.array(args.field)
        with (args.output / "fields.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["tau", "x", "y", "unit_step_temperature"])
            for t in np.unique(points[:, 0]):
                xy = points[points[:, 0] == t, 1:]
                writer.writerows((t, x, y, v) for (x, y), v in zip(xy, op.fields(t, xy)))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
