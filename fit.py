"""A single rational readout trained on the differential equation for a sustained heat step."""

from pathlib import Path
from time import perf_counter
import argparse
import json
import math
import numpy as np
import torch


def design(x, z, derivative):
    def response(p):
        d = x[:, None] + p
        return x[:, None] / d + (p / d**2 if derivative else 0.0)

    pos, neg = response(z), response(z.conj())
    return torch.cat(((pos + neg) / 2, 1j * (pos - neg) / 2), dim=1)


def constrained(D, target, N):
    # Eliminate the final real residue so R(infinity)=1 exactly.
    offset = D[:, N - 1]
    basis = torch.cat((D[:, : N - 1] - offset[:, None], D[:, N:]), 1)
    rhs = target - offset
    return torch.cat((basis.real, basis.imag)), torch.cat((rhs.real, rhs.imag))


def expand(y, N):
    return torch.cat((y[: N - 1], (1 - y[: N - 1].sum()).reshape(1), y[N - 1 :]))


def samples(device, n, seed):
    rng = np.random.default_rng(seed)
    x = np.exp(rng.uniform(-7.5, 11, n) + 1j * rng.uniform(0, math.radians(75), n))
    return torch.tensor(x, device=device)


def train(destination, count=10, device="cpu", steps=800):
    N = count
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    x = samples(device, 1600, 19039)
    valid = samples(device, 4096, 85233)
    rate = torch.nn.Parameter(torch.zeros((2, N), device=device))
    freq = torch.arange(N, device=device) + 0.5
    opt = torch.optim.Adam([rate], lr=0.025)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=0.0004)
    ridge = 8e-7
    ridge_matrix = ridge * torch.eye(2 * N - 1, device=device)

    def poles():
        return 3.5 * torch.exp(1.8 * torch.tanh(rate[0])) + 1j * np.pi * freq * torch.exp(
            0.45 * torch.tanh(rate[1])
        )

    def coefficients(z):
        with torch.no_grad():
            A, rhs = constrained(design(x, z, True), torch.ones_like(x), N)
            M = torch.cat((A, ridge_matrix))
            y = torch.linalg.lstsq(
                M, torch.cat((rhs, torch.zeros(2 * N - 1, device=device))), driver="gels"
            ).solution
            return y

    def export(label, z, y):
        c = expand(y, N)
        weights = (c[:N] + 1j * c[N:]).detach().cpu().numpy()
        np.savez(destination / (label + ".npz"), z=z.detach().cpu().numpy(), w=weights)
        res = design(valid, z, True) @ c.to(torch.complex128) - 1.0
        error = design(valid, z, False) @ c.to(torch.complex128) + torch.expm1(-valid)
        return dict(
            physics_rms=float(res.abs().square().mean().sqrt()),
            modal_max=float(error.abs().max()),
            modal_rms=float(error.abs().square().mean().sqrt()),
            weight_norm=float(np.linalg.norm(weights)),
        )

    start = perf_counter()
    records = {}
    for step in range(steps + 1):
        z = poles()
        y = coefficients(z)
        if step in (0, steps):
            label = "fixed" if step == 0 else "fitted"
            with torch.no_grad():
                records[label] = export(label, z, y)
        if step == steps:
            break
        A, rhs = constrained(design(x, z, True), torch.ones_like(x), N)
        obj = (A @ y - rhs).square().sum() + ridge**2 * y.square().sum()
        opt.zero_grad()
        torch.log(obj).backward()
        opt.step()
        sched.step()
    records.update(
        device=torch.cuda.get_device_name() if device == "cuda" else device,
        steps=steps,
        seconds=perf_counter() - start,
    )
    (destination / "training.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/fit"))
    parser.add_argument("--count", type=int, choices=range(4, 13), default=10)
    args = parser.parse_args()
    train(args.output, args.count, args.device)
