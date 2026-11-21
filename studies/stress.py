"""Measure spatial self-convergence at higher flow and smaller wall gaps."""

import numpy as np

from studies.convergence import CASES, RULES, TIMES, outputs, read_rule, rms, save


def geometries():
    for pe in (CASES["Q421"]["case"]["Pe"], 10, 12, 15):
        case = CASES["Q421"]["case"].copy()
        case["Pe"] = pe
        yield "flow", "Q421", pe, case
    parent = CASES["Q311"]["case"]
    original_gap = 1 - np.hypot(parent["xp1"], parent["yp1"]) - parent["Rp1"]
    for gap in (original_gap, 0.02, 0.01, 0.005):
        case = CASES["Q311"]["case"].copy()
        center = np.array([case["xp1"], case["yp1"]])
        center *= (1 - case["Rp1"] - gap) / np.linalg.norm(center)
        case["xp1"], case["yp1"] = center
        assert np.isclose(1 - np.linalg.norm(center) - case["Rp1"], gap)
        assert np.linalg.norm(center - [case["xp2"], case["yp2"]]) > case["Rp1"] + case["Rp2"]
        yield "gap", "Q311", gap, case


def main():
    rule = read_rule(RULES / "fitted.npz")
    levels = [(12, 22, 28), (18, 32, 42), (24, 42, 56)]
    records = []
    for kind, name, value, case in geometries():
        vals = [outputs(case, res, TIMES, rule) for res in levels]
        for idx in range(2):
            for (time, observable), result in vals[idx].items():
                target = vals[-1][time, observable]
                delta = rms(result - target)
                records.append(
                    dict(
                        parent=name,
                        varied=kind,
                        value=value,
                        level=idx,
                        tau=time,
                        observable=observable,
                        rms=delta,
                        relative_pct=100 * delta / rms(target),
                    )
                )
    save("stress.csv", records)


if __name__ == "__main__":
    main()
