"""Manuscript assessment layouts, physical probe coordinates and time grids."""

import csv
import json
from pathlib import Path
import numpy as np
from fea.reference import local_probes, extended_probes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RULES = DATA / "rules"
PLAN = json.loads((DATA / "single_u.json").read_text())
CASES = PLAN["cases"]
OPERATING = PLAN["signed_cases"]
GRID = {
    "time_edges": [0, 0.02, 0.1, 1, 2, 10, 20, 56, 100],
    "coarse_steps": [0.001, 0.004, 0.02, 0.04, 0.08, 0.1, 0.24, 0.4],
    "snapshots": [0, 0.02, 1, 10, 20, 100],
    "wake_times": [10, 20, 100],
}


def save_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def command(t):
    values = np.take(PLAN["levels"], np.searchsorted(PLAN["switches"], np.asarray(t) - 1e-10))
    return np.where(np.asarray(t) == 0.0, 0.0, values)


def time_nodes(factor=2):
    edges = GRID["time_edges"]
    intervals = [
        np.linspace(a, b, round((b - a) / dt) * factor + 1)[:-1]
        for a, b, dt in zip(edges[:-1], edges[1:], GRID["coarse_steps"], strict=True)
    ]
    return np.r_[np.concatenate(intervals), edges[-1]]


def probes(case):
    aligned, physical = extended_probes(case)
    return local_probes(case), physical[aligned[:, 0] >= 3]
