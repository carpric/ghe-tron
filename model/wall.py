"""Geometry-independent deterministic impedance interpolation for fluid-only frequencies."""

from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import BarycentricInterpolator
from .boundary import Exterior


class WallTable:
    def __init__(self, directory, wall, radial, padding):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True, parents=True)
        self.wall, self.radial, self.padding = wall, radial, padding
        self.m = np.arange(-wall, wall + 1)
        self.eye = np.eye(len(self.m))
        self.poles = np.geomspace(0.035, 2500, 30)

    def features(self, s):
        p = np.sqrt(s)
        return np.r_[1.0, p / (self.poles + p), s / (self.poles[::2] ** 2 + s)]

    def fit_node(self, pe):
        path = self.directory / f"joint_p{pe:.4f}_w{self.wall}_r{self.radial}_a{self.padding}.npz"
        if not path.exists():
            start = perf_counter()
            ext = Exterior(pe, self.m, self.radial, self.padding)
            freqs = (np.geomspace(0.03, 5e5, 36)[:, None] * np.exp(1j * np.array([0.0, 0.8, 1.54]))).ravel()
            X = np.array([self.features(s) for s in freqs])
            values = np.array([(ext.response(s, False)[0] - np.sqrt(s) * self.eye).ravel() for s in freqs])
            coefficients = np.linalg.lstsq(
                np.vstack([X.real, X.imag]), np.vstack([values.real, values.imag]), rcond=1e-13
            )[0]
            np.savez_compressed(path, coeff=coefficients, seconds=perf_counter() - start)
        with np.load(path) as data:
            return data["coeff"]

    def adjacent(self, pe):
        # Each unit-Pe interval has a fixed degree-four table; geometry never enters it.
        lower = min(9, int(pe))
        return lower + np.arange(5) * 0.25

    def prepare(self, pe):
        for x in self.adjacent(pe):
            self.fit_node(x)

    def attach(self, model):
        pe = model.par["Pe"]
        grid = self.adjacent(pe)
        coefficients = BarycentricInterpolator(grid, np.array([self.fit_node(x) for x in grid]))(pe)
        direct = model.exterior.response

        def evaluate(s, need_field=True):
            if need_field:
                return direct(s, True)
            value = (self.features(s) @ coefficients).reshape(self.eye.shape)
            return value + np.sqrt(s) * self.eye, None

        model.exterior.response = evaluate
