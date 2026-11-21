"""Fixed comparison inputs and observation grid, in borehole-radius units."""

from studies.inputs import RESULTS
import numpy as np

HERE = RESULTS / "circuits"
CENTRES = np.array([[-0.36, 0.36], [0.36, 0.36], [0.36, -0.36], [-0.36, -0.36]])
RADII = np.full(4, 0.145)
BETA = np.full(4, 0.16)
COUPLING = 0.55
LAMBDA, GAMMA, ANGLE = 1.2, 0.8, np.deg2rad(25.0)
PAIRS = {"adjacent": [(0, 3), (1, 2)], "opposite": [(0, 3), (2, 1)]}
SNAP = np.array([0.02, 1.0, 10.0, 20.0, 100.0])
# Each tuple specifies supply layout, Péclet number, and first-branch flow fraction.
CASES = [(kind, pe, 0.5) for pe in (0.0, 2.0, 5.0) for kind in PAIRS] + [("opposite", 2.0, 0.35)]


def name(kind, pe, fraction):
    return f"{kind}_pe{pe:g}_split{round(100*fraction)}"


def times(level):
    edges = [0.0, 0.02, 0.1, 1.0, 2.0, 10.0, 20.0, 56.0, 100.0]
    steps = [0.001, 0.004, 0.02, 0.04, 0.08, 0.1, 0.24, 0.4]
    return np.r_[
        np.concatenate(
            [
                np.linspace(lo, hi, round((hi - lo) / step) * level + 1)[:-1]
                for lo, hi, step in zip(edges, edges[1:], steps)
            ]
        ),
        100.0,
    ]


def signed_command(time):
    """Positive interval ending at 5, cooling through 12, then reheating."""
    return np.select([time == 0, time <= 5.0 + 1e-12, time <= 12.0 + 1e-12], [0.0, 1.0, -0.7], default=0.35)


def probes(centres=CENTRES, radii=RADII, angle=ANGLE):
    axis = np.linspace(-3, 3, 61)
    x, y = np.meshgrid(axis, axis)
    xy = np.c_[x.ravel(), y.ravel()]
    keep = (np.linalg.norm(xy[:, None] - centres, axis=2) > radii + 0.001).all(axis=1)
    local = xy[keep]
    x, y = np.meshgrid(np.arange(3.0, 25.01, 0.5), np.arange(-8.0, 8.01, 0.5))
    aligned = np.c_[x.ravel(), y.ravel()]
    c, s = np.cos(angle), np.sin(angle)
    return local, aligned @ np.array([[c, s], [-s, c]])
