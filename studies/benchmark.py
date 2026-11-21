"""Measure precomputation, one joint prediction, and causal FFT throughput separately."""

from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
from scipy.interpolate import CubicSpline
from model.boundary import BoundaryOperator
from model.response import ANCHORS, FIELD_TIMES
from model.loads import reconstruct
from model.quadrature import nodes
from model.wall import WallTable
from studies.inputs import CASES, OPERATING, RESULTS, save_csv, time_nodes
from studies.single import predict, read_rule


def fluid_interpolator(case, table):
    """Build the scalar inlet readout used for the timed single-U fluid workload."""
    model = BoundaryOperator(case, 12, 22, 28, 14)
    if case["Pe"]:
        table.attach(model)
    z, w = read_rule()
    means = [
        sum((weight * model.frequency(pole / t, t in FIELD_TIMES)[0]).real for pole, weight in zip(z, w))
        for t in ANCHORS
    ]
    spline = CubicSpline(np.log(ANCHORS), means)

    def fluid(times):
        inlet = np.full(times.shape, model.right)
        inlet[1:] += spline(np.log(times[1:]))
        return np.column_stack((inlet, inlet - 1))

    return fluid


def run():
    freq, weight = nodes([1.0], A=14.0, n=12, m=12)
    rules = {"fixed": read_rule("fixed.npz"), "trained": read_rule(), "table": (freq[0], weight[0] / freq[0])}
    rng = np.random.default_rng(6788322)
    with TemporaryDirectory() as folder:
        from pathlib import Path

        cache = Path(folder)
        table = WallTable(cache, 22, 28, 14)
        start = perf_counter()
        for layout in CASES.values():
            if layout["case"]["Pe"] > 0:
                table.prepare(layout["case"]["Pe"])
        files = list(cache.glob("*.npz"))
        save_csv(
            RESULTS / "table_setup.csv",
            [
                dict(
                    seconds=perf_counter() - start,
                    nodes=len(files),
                    bytes=sum(f.stat().st_size for f in files),
                )
            ],
        )
        tasks = [(name, arm) for name in CASES for arm in rules]
        rows = []
        for idx in rng.permutation(len(tasks)):
            name, arm = tasks[idx]
            start = perf_counter()
            predict(CASES[name]["case"], rules[arm], table)
            rows.append(dict(layout=name, arm=arm, seconds=perf_counter() - start))
        save_csv(RESULTS / "quiet_times.csv", rows)
        t = np.linspace(0, 100, 2001)
        holds = np.random.default_rng(29741).uniform(-1, 1, (1000, 100))
        loads = np.column_stack((np.zeros(len(holds)), np.repeat(holds, 20, axis=1)))
        rows = []
        for name in OPERATING:
            for repeat in range(3):
                start = perf_counter()
                fluid_at = fluid_interpolator(CASES[name]["case"], table)
                fluid_at(t)
                elapsed = perf_counter() - start
                start = perf_counter()
                n = len(t) - 1
                nfft = next_fast_len(2 * n - 1)
                base = time_nodes(2)
                kernel = CubicSpline(base, fluid_at(base), axis=0)(t)
                transformed = rfft(np.diff(loads, axis=1), nfft, axis=1)
                # The unit-step inlet/outlet difference is one at every time.
                inlet = irfft(transformed * rfft(kernel[1:, 0], nfft), nfft, axis=1)[:, :n]
                fluid = np.zeros((len(loads), len(t), 2))
                fluid[:, 1:, 0] = inlet
                fluid[:, 1:, 1] = inlet - loads[:, 1:]
                seconds = perf_counter() - start
                for k in (0, 278, 999):
                    np.testing.assert_allclose(fluid[k], reconstruct(kernel, loads[k]), atol=1e-11, rtol=0)
                rows.append(
                    dict(
                        layout=name,
                        rep=repeat,
                        fluid_kernel_seconds=elapsed,
                        batch_seconds=seconds,
                        histories=len(loads),
                        samples=len(t),
                    )
                )
        save_csv(RESULTS / "batch_times.csv", rows)
