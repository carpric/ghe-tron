"""Strong-flow wake stationarity and sensitivity to the FE exterior radius."""

import numpy as np

from model.wall import WallTable
from studies.inputs import CASES, GRID, RESULTS, save_csv
from studies.unit import TABLES, calculate_reference, store, unit_step_responses


def relative_change_pct(values, reference):
    return 100 * np.linalg.norm(values - reference) / np.linalg.norm(reference)


def radius_comparison(name, standard):
    case = CASES[name]["case"]
    enlarged = store(name, "fine_radius75", lambda: calculate_reference(case, 2, radius=75)[0])
    times = enlarged["tau"]
    np.testing.assert_allclose(standard["tau"], times, rtol=0, atol=1e-12)
    error = np.trapezoid((standard["fluid"] - enlarged["fluid"]) ** 2, times, axis=0)
    signal = np.trapezoid(enlarged["fluid"] ** 2, times, axis=0)
    rows = []
    for leg, value in zip(("supply", "return"), 100 * np.sqrt(error / signal), strict=True):
        rows.append(dict(layout=name, observable=leg, upper_or_snapshot_time=times[-1],
                         difference_relative_to_radius75_pct=value))
    for t, small, large in zip(GRID["wake_times"], standard["wake"], enlarged["wake"], strict=True):
        rows.append(dict(layout=name, observable="wake", upper_or_snapshot_time=t,
                         difference_relative_to_radius75_pct=relative_change_pct(small, large)))
    return rows


def run():
    table = WallTable(TABLES, 22, 28, 14)
    early, late = (GRID["wake_times"].index(t) for t in (20, 100))
    stationarity = {"multipole": [], "fine_FE": [], "coarse_FE": []}
    for name in sorted(k for k, layout in CASES.items() if layout["flow"] == "strong"):
        operator, coarse, fine = unit_step_responses(name, table)
        for label, response in (("multipole", operator), ("fine_FE", fine), ("coarse_FE", coarse)):
            wake = response["wake"]
            change = relative_change_pct(wake[early], wake[late])
            stationarity[label].append(dict(
                layout=name, method=label, wake_change_20_to_100_relative_to_100_pct=change
            ))
        if name == "Q421":
            save_csv(RESULTS / "verification_radius.csv", radius_comparison(name, fine))
        print(f"Checked wake for {name}", flush=True)
    save_csv(RESULTS / "wake_stationarity.csv", [row for rows in stationarity.values() for row in rows])
