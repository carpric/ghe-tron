"""Independent volume and boundary quadrature for the grout energy balance."""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ive, kve
from model.boundary import BoundaryOperator
from model.quadrature import nodes
from studies.inputs import CASES, RESULTS, probes, save_csv
from studies.single import read_rule, predict


def volume_rule(model, degree):
    g, w = leggauss(degree)
    cut = [0.0, 2 * np.pi]
    for c, r in zip(model.centres, model.radii):
        if abs(c) > r:
            half = np.arcsin(r / abs(c))
            cut.extend([(np.angle(c) + sign * half) % (2 * np.pi) for sign in (-1, 1)])
    cuts = np.unique(cut)
    nodes, weights = [], []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        # The ray intersects pipe circles at square-root endpoints. Smooth both tangencies.
        t = (g + 1) * np.pi / 2
        angle = lo + (hi - lo) * (1 - np.cos(t)) / 2
        weight = w * np.pi / 4 * (hi - lo) * np.sin(t)
        for theta, wt in zip(angle, weight):
            direction = np.exp(1j * theta)
            limits = [0.0, 1.0]
            for c, r in zip(model.centres, model.radii):
                z = c / direction
                if abs(z.imag) < r:
                    d = np.sqrt(r * r - z.imag**2)
                    limits.extend(t for t in (z.real - d, z.real + d) if 0 < t < 1)
            limits.sort()
            for a, b in zip(limits[:-1], limits[1:]):
                if any(abs((a + b) / 2 * direction - c) < r for c, r in zip(model.centres, model.radii)):
                    continue
                radius = (a + b) / 2 + (b - a) * g / 2
                nodes.extend(radius * direction)
                weights.extend(w * wt * (b - a) / 2 * radius)
    return np.array(nodes), np.array(weights)


def audit(model, s, nodes, weights):
    inc, state = model.frequency(s, True)
    g, reg, h, wall, _, ib, pipe = state

    def temperature(z):
        r = abs(z)
        temp = np.sum(
            ive(abs(model.m), g * r[:, None])
            / ib[abs(model.m)]
            * reg
            * np.exp(g.real * (r - 1)[:, None] + 1j * np.angle(z)[:, None] * model.m),
            axis=1,
        )
        for c, r0, P, local in zip(model.centres, model.radii, pipe, np.split(h, 2)):
            shifted = z - c
            r = abs(shifted)
            temp += np.sum(
                kve(abs(model.n), g * r[:, None])
                / P[1][abs(model.n)]
                * local
                * np.exp(-g * (r - r0)[:, None] + 1j * np.angle(shifted)[:, None] * model.n),
                axis=1,
            )
        return temp

    direction = np.exp(2j * np.pi * np.arange(192) / 192)
    eps = 2e-6

    def derivative(c, r):
        return (temperature(c + (r + eps) * direction) - temperature(c + (r - eps) * direction)) / (2 * eps)

    lam = model.par["lam"]
    gamma = model.par["gamma"]
    flux = model.exterior.response(s, False)[0] @ wall
    wall_flux = np.exp(1j * np.angle(direction)[:, None] * model.m) @ flux
    wall_error = lam * derivative(0, 1) + wall_flux
    energies = []
    pipe_rms = []
    for j, (c, r, beta) in enumerate(zip(model.centres, model.radii, model.beta)):
        grad = derivative(c, r)
        pipe_rms.append(
            np.sqrt(np.mean(abs(temperature(c + r * direction) - beta * grad - (model.right + inc - j)) ** 2))
        )
        energies.append(-2 * np.pi * lam * r * grad.mean())
    solid_integral = weights @ temperature(nodes)
    supplied = sum(energies)
    C = model.psi[0] * model.beta[0] / model.radii[0]
    input_error = abs(C * supplied / (2 * np.pi * lam) - 1)
    storage = lam * gamma * s * solid_integral
    imbalance = abs(storage + 2 * np.pi * wall_flux.mean() - supplied) / abs(supplied)
    return dict(
        pipe_input_error=input_error,
        max_pipe_robin_rms=max(pipe_rms),
        interface_flux_rms=np.sqrt(np.mean(abs(wall_error) ** 2)),
        relative_grout_energy_defect=imbalance,
    )


def run():
    energies, rotations = [], []
    for name, layout in CASES.items():
        case = layout["case"]
        model = BoundaryOperator(case, 12, 22, 28, 14)
        points, weights = volume_rule(model, 24)
        for s in (0.07 + 0.03j, 7 + 3j, 350 + 150j):
            energies.append(dict(case=name, s_real=s.real, s_imag=s.imag, **audit(model, s, points, weights)))
        if case["Pe"]:
            continue
        xy, down = probes(case)
        angle = 0.873
        cs, sn = np.cos(angle), np.sin(angle)
        rot = np.array([[cs, -sn], [sn, cs]])
        changed = dict(case, phi_v=2.48)
        for k in (1, 2):
            changed[f"xp{k}"], changed[f"yp{k}"] = rot @ [case[f"xp{k}"], case[f"yp{k}"]]
        wall = np.abs(np.linalg.norm(xy, axis=1) - 1) < 1e-12
        z, w = nodes([1.0], A=14.0, n=12, m=12)
        for arm, rule in [("optimized", read_rule()), ("table", (z[0], w[0] / z[0]))]:
            a = predict(case, rule)
            b = predict(changed, rule, locations=(xy @ rot.T, down @ rot.T))
            rotations.append(
                dict(
                    case=name,
                    arm=arm,
                    fluid_sup=np.max(abs(a["fluid"] - b["fluid"])),
                    solids_away_from_exact_wall_sup=np.max(abs(a["local"][:, ~wall] - b["local"][:, ~wall])),
                    exact_interface_sup=np.max(abs(a["local"][:, wall] - b["local"][:, wall])),
                )
            )
    save_csv(RESULTS / "energy.csv", energies)
    save_csv(RESULTS / "rotation.csv", rotations)
