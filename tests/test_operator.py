"""Fast checks of the public entry point against archived predictions and circuit identities."""

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from infer import bind, read_layout
from model.loads import reconstruct
from studies.inputs import DATA, probes, CASES


class OperatorTests(unittest.TestCase):
    def test_single_u_at_reported_nodes(self):
        with TemporaryDirectory() as cache, np.load(DATA / "figure_samples.npz") as archive:
            model = bind(read_layout("Q222"), Path(cache))
            time = archive["Q222_trained_tau"]
            np.testing.assert_allclose(model.fluid(time), archive["Q222_trained_fluid"], atol=1e-9, rtol=0)
            # Use the full published probe set so both calculations have the same radial interpolation range.
            local, wake = probes(CASES["Q222"]["case"])
            values = model.fields(10, np.vstack((local, wake)))
            expected = np.r_[archive["Q222_trained_ground10"], archive["Q222_trained_wake10"]]
            np.testing.assert_allclose(values, expected, atol=1e-8, rtol=0)

    def test_parallel_circuit_and_load_changes(self):
        layout = read_layout("T01")
        with TemporaryDirectory() as cache:
            model = bind(layout, Path(cache))
            time = np.arange(5, dtype=float)
            load = np.array([0.0, 1.0, 1.0, -0.5, 0.25])
            output = model.circuit_temperatures(time, load)
            np.testing.assert_allclose(output["supply"] - output["mixed_return"], load, atol=1e-12)
            np.testing.assert_allclose(output["returns"] @ layout["fractions"], output["mixed_return"])
            kernel = model.fluid(time)
            by_steps = np.zeros_like(kernel)
            for j, delta in enumerate(np.diff(load), start=1):
                by_steps[j:] += delta * kernel[1 : len(time) - j + 1]
            np.testing.assert_allclose(reconstruct(kernel, load), by_steps, atol=1e-12)
