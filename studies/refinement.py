"""Compare the three reference levels on nested times and identical field probes."""

import csv
import json

import numpy as np

from studies.inputs import RESULTS

HERE = RESULTS / "verification"
CASES = ("Q122", "Q222", "Q322", "Q422")
LOCAL_TIMES = (0, 0.02, 1, 10, 20, 100)
WAKE_TIMES = (10, 20, 100)


def write_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def observations(reference, operator):
    t = reference[0]["tau"]
    time_series = []
    for data, stride in zip([*reference, operator], (1, 2, 4, 2)):
        np.testing.assert_allclose(t, data["tau"][::stride], atol=1e-12, rtol=0)
        time_series.append(data["fluid"][::stride])
    for col, observable in enumerate(("inlet", "outlet")):
        for horizon in (0.02, 2.0, 100.0):
            mask = t <= horizon + 1e-13
            nodes = t[mask]
            increments = np.diff(nodes)
            weights = (np.r_[0, increments] + np.r_[increments, 0]) / (2 * horizon)
            yield observable, horizon, [x[mask, col] for x in time_series], weights
    grout = operator["grout"]
    for col, time in enumerate(LOCAL_TIMES[1:], start=1):
        for name, mask in (("grout", grout), ("ground", ~grout)):
            yield name, time, [a["local"][col, mask] for a in [*reference, operator]], None
    if reference[0]["wake"].shape[1]:
        for col, time in enumerate(WAKE_TIMES):
            yield "wake", time, [a["wake"][col] for a in [*reference, operator]], None


def compare(values, weights):
    coarse, fine, third, model = values

    def inner(x, y):
        product = x * y
        return np.mean(product) if weights is None else np.dot(weights, product)

    def norm(x):
        return np.sqrt(inner(x, x))

    first, second = fine - coarse, third - fine
    d12, d23 = norm(first), norm(second)
    order = np.log2(d12 / d23)
    cosine = inner(first, second) / (d12 * d23)
    resolved = 0.5 <= order <= 3 and cosine >= 0.90
    scale = norm(third)
    result = dict(
        rms_FE3=scale,
        coarse_vs_FE3_pct=100 * norm(coarse - third) / scale,
        fine_vs_FE3_pct=100 * d23 / scale,
        model_vs_FE3_pct=100 * norm(model - third) / scale,
        model_vs_FE2_pct=100 * norm(model - fine) / norm(fine),
        delta12_rms=d12,
        delta23_rms=d23,
        p=order,
        cosine=cosine,
        extrapolation_used=bool(resolved),
        model_vs_extrap_pct="",
    )
    if resolved:
        extrapolated = third + second / (2**order - 1)
        result["model_vs_extrap_pct"] = 100 * norm(model - extrapolated) / norm(extrapolated)
    return result


def summarize(names=CASES):
    raw = HERE / "raw"
    records, times = [], []
    for name in names:
        levels = [np.load(raw / f"{name}_{method}.npz") for method in ("1", "2", "3", "operator")]
        for observable, time, values, weights in observations(levels[:3], levels[-1]):
            row = compare(values, weights)
            records.append(dict(case=name, observable=observable, tau=time, **row))
        for data in levels:
            data.close()
        for method in ("1", "2", "3", "operator"):
            meta = json.loads((raw / f"{name}_{method}.json").read_text())
            repeats = meta.get("repeats", [""] * 3)
            times.append(
                dict(
                    case=name,
                    method=method,
                    dofs=meta.get("dofs", ""),
                    seconds=meta["seconds"],
                    first_use_s=meta.get("first", ""),
                    table_setup_s=meta.get("setup", ""),
                    repeat1=repeats[0],
                    repeat2=repeats[1],
                    repeat3=repeats[2],
                )
            )
    return records, times


import pandas as pd
from scipy.interpolate import CubicSpline


def direct_check():
    anchors = pd.read_csv(HERE / "fluid_anchors.csv")
    records = []
    for name in CASES:
        a = anchors[anchors.case == name]
        with (
            np.load(HERE / "raw" / f"{name}_3.npz") as fe,
            np.load(HERE / "raw" / f"{name}_operator.npz") as operator,
        ):
            times = fe["tau"][::4]
            obs = fe["fluid"][::4, 0]
            original = operator["fluid"][::2, 0]
            initial = original[0]

            def interpolate(col):
                return np.r_[initial, CubicSpline(np.log(a.tau), a[col])(np.log(times[1:]))]

            cached = interpolate("production_inlet")
            direct = interpolate("direct_inlet")
            np.testing.assert_allclose(cached, original, atol=2e-10, rtol=1e-10)
            for label, values in [
                ("tabulated", original),
                ("direct", direct),
                ("exact_pe_fit", interpolate("exact_pe_fit")),
            ]:
                relative = 100 * np.sqrt(
                    np.trapezoid((values - obs) ** 2, times) / np.trapezoid(obs**2, times)
                )
                records.append(dict(case=name, wall=label, inlet_full_vs_third_pct=relative))
    write_csv(HERE / "wall_check.csv", records)
    return pd.DataFrame(records)
