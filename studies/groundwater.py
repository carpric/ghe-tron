"""Fixed-layout Darcy-flux sensitivity, in the dimensional Q221 example."""

import csv
from tempfile import TemporaryDirectory

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from studies.inputs import CASES, RESULTS, GRID, local_probes, time_nodes
from studies.single import predict, read_rule
from model.wall import WallTable
from studies.unit import calculate_reference

SNAPSHOTS = [10.0, 100.0]
RADIUS = 0.075
GROUND_CONDUCTIVITY = 2.2
WATER_CAPACITY = 4.18e6
DIFFUSIVITY = 1.0e-6
INITIAL_C = 12.0


def probe_pair(xy, values, direction):
    """Bilinear interpolation of both solutions on the same physical probe grid."""
    axis = np.linspace(-3, 3, 121)
    ix, iy = np.rint((xy + 3) * 20).astype(int).T
    field = np.full((len(axis), len(axis)), np.nan)
    field[iy, ix] = values
    points = np.array([-2.0, 2.0])[:, None] * direction
    return RegularGridInterpolator((axis, axis), field)(points[:, ::-1])


def solutions(case, table_dir):
    idx = [GRID["snapshots"].index(t) for t in SNAPSHOTS]
    op = predict(case, read_rule(), WallTable(table_dir, 22, 28, 14))
    ref, _ = calculate_reference(case, 2)
    for data in (op, ref):
        yield data["fluid"], data["local"][idx]


def main():
    case = CASES["Q221"]["case"]
    coupling = case["psi1"] * case["beta1"] / case["Rp1"]
    scale = 50.0 * coupling / (2 * np.pi * case["lam"] * GROUND_CONDUCTIVITY)
    time = time_nodes(2)
    days = time * RADIUS**2 / DIFFUSIVITY / 86400.0
    direction = np.array([np.cos(case["phi_v"]), np.sin(case["phi_v"])])
    xy = local_probes(case)
    histories, probes = [], []
    with TemporaryDirectory() as tables:
        for pe in (0.0, case["Pe"], 1.0, 5.0):
            flux = pe * GROUND_CONDUCTIVITY / (RADIUS * WATER_CAPACITY) * 86400.0
            print(f"Evaluating fixed layout Q221 at Pe={pe:g}", flush=True)
            pair = solutions(dict(case, Pe=pe), tables)
            for source, (hist, fields) in zip(("GHE-TRON", "fine FE"), pair, strict=True):
                np.testing.assert_allclose(hist[:, 0] - hist[:, 1], 1.0, atol=1e-9, rtol=0)
                temperatures = INITIAL_C + scale * hist
                if not np.isfinite(temperatures).all():
                    raise ValueError("Nonfinite fluid temperatures")
                histories.extend(
                    (pe, flux, source, t, d, ti, to)
                    for t, d, (ti, to) in zip(time, days, temperatures, strict=True)
                )
                for t, field in zip(SNAPSHOTS, fields, strict=True):
                    up, down = scale * probe_pair(xy, field, direction)
                    if not np.isfinite([up, down]).all():
                        raise ValueError("Incomplete ground probe interpolation")
                    probes.append((pe, flux, source, t, up, down))
    for name, headings, records in (
        ("groundwater_fluids", "Pe darcy_m_per_day source tau time_days inlet_C outlet_C", histories),
        ("groundwater_probes", "Pe darcy_m_per_day source tau upstream_rise_K downstream_rise_K", probes),
    ):
        with (RESULTS / (name + ".csv")).open("w", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(headings.split())
            writer.writerows(records)


if __name__ == "__main__":
    main()
