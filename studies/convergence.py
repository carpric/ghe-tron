"""Matched direct-resolvent inversion using the Dingfelder-Weideman contour."""

import numpy as np

from studies.inputs import CASES, OPERATING as CASES4, RULES, RESULTS, save_csv
from studies import single
from model.quadrature import nodes as euler

TIMES = (0.02, 10.0, 100.0)
BASE = (12, 22, 28)


def save(name, rows):
    save_csv(RESULTS / "verification" / name, rows)


def talbot(j):
    """Use J positive-imaginary midpoints on the symmetric 2J-node contour."""
    theta = np.pi * (np.arange(j) + 0.5) / j
    angle_scale = 0.6407
    cot = 1 / np.tan(angle_scale * theta)
    z = 2 * j * (-0.6122 + 0.5017 * theta * cot + 0.2645j * theta)
    dz = 2 * j * (0.5017 * (cot - angle_scale * theta / np.sin(angle_scale * theta) ** 2) + 0.2645j)
    return z, np.exp(z) * dz / (1j * j * z)


def euler_rule(j):
    order = (j - 1) // 2
    s, w = euler([1.0], A=18.0 if j == 45 else 16.0, n=order, m=order)
    return s[0], w[0] / s[0]


def read_rule(path):
    with np.load(path) as values:
        return values["z"], values["w"]


def outputs(case, resolution, times, rule):
    model = single.BoundaryOperator(case, *resolution, 14)
    readout = single.Readout(model, 96)
    nodes, weights = rule
    results = {}
    for time in times:
        coefficients = None
        supply = model.right
        for frequency, weight in zip(nodes / time, weights):
            increment, state = model.frequency(frequency, True)
            supply += (weight * increment).real
            profiles = model.profiles(frequency, state, readout.grid)
            if coefficients is None:
                coefficients = [weight * v for v in profiles]
            else:
                for accumulated, v in zip(coefficients, profiles):
                    accumulated += weight * v
        local, wake = readout.field(coefficients)
        results[time, "fluid"] = np.array([supply, supply - 1])
        results[time, "grout"] = local[readout.grout]
        results[time, "ground"] = local[~readout.grout]
        if time >= 10 and case["Pe"]:
            results[time, "wake"] = wake
    if not all(np.isfinite(v).all() for v in results.values()):
        raise ArithmeticError(f"Nonfinite response at spatial resolution {resolution}")
    return results


def rms(v):
    return np.sqrt(np.mean(v * v))


def curves():
    weights = np.load(RULES / "order_sweep.npz")
    rows = []
    for name in CASES4:
        ref = outputs(CASES[name]["case"], BASE, TIMES, euler_rule(45))
        check = outputs(CASES[name]["case"], BASE, TIMES, euler_rule(37))
        for j in range(4, 13):
            fitted = weights[f"optimized_z{j}"], weights[f"optimized_w{j}"]
            for method, rule in (("Talbot", talbot(j)), ("optimized", fitted)):
                values = outputs(CASES[name]["case"], BASE, TIMES, rule)
                for (time, observable), target in ref.items():
                    scale = rms(target)
                    error = rms(values[time, observable] - target)
                    rows.append(
                        dict(
                            case=name,
                            tau=time,
                            observable=observable,
                            J=j,
                            method=method,
                            rms=error,
                            relative_pct=100 * error / scale,
                            ref_rms=scale,
                            reference_difference=rms(check[time, observable] - target),
                        )
                    )
        print("Compared", name, flush=True)
    weights.close()
    save("inversion.csv", rows)


def spatial():
    weights = read_rule(RULES / "fitted.npz")
    bounds = []
    for name, layout in CASES.items():
        c = layout["case"]
        xy = np.array([[c[f"xp{k}"], c[f"yp{k}"]] for k in (1, 2)])
        rad = np.array([c["Rp1"], c["Rp2"]])
        bounds.append(
            dict(
                case=name,
                pe=c["Pe"],
                pipe_gap=np.linalg.norm(xy[0] - xy[1]) - sum(rad),
                wall_gap=min(1 - np.linalg.norm(xy, axis=1) - rad),
            )
        )
    save("single_envelope.csv", bounds)
    tight = min(bounds, key=lambda r: min(r["pipe_gap"], r["wall_gap"]))["case"]
    high_pe = max(bounds, key=lambda r: r["pe"])["case"]
    low_pe = min((r for r in bounds if r["pe"] > 0), key=lambda r: r["pe"])["case"]
    sweeps = [
        ("pipe", tight, (8, 12, 16, 20, 24)),
        ("wall", high_pe, (14, 22, 28, 34, 40)),
        ("radial", low_pe, (16, 28, 42, 56, 84)),
    ]
    rows = []
    for index, (dimension, name, levels) in enumerate(sweeps):
        values = []
        for level in levels:
            resolution = list(BASE)
            resolution[index] = level
            values.append(outputs(CASES[name]["case"], resolution, TIMES, weights))
        reference = values[-1]
        for level, data in zip(levels, values):
            for (time, observable), target in reference.items():
                error = rms(data[time, observable] - target)
                rows.append(
                    dict(
                        case=name,
                        dimension=dimension,
                        level=level,
                        reference_level=levels[-1],
                        tau=time,
                        observable=observable,
                        rms=error,
                        relative_pct=100 * error / rms(target),
                    )
                )
    save("spatial.csv", rows)
