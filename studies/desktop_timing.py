"""Serial error/work comparison on the four operating layouts."""

from time import perf_counter
import json
import numpy as np
from studies.inputs import CASES, OPERATING, RESULTS, save_csv
from studies.unit import calculate_reference
from studies.single import predict, read_rule
from model.wall import WallTable
from studies.refinement import summarize, direct_check


def time_operator(name, raw):
    rule = read_rule()
    pe = CASES[name]["case"]["Pe"]
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(dir=raw) as cache:
        tick = perf_counter()
        table = WallTable(cache, 22, 28, 14)
        if pe:
            table.prepare(pe)
        setup = perf_counter() - tick
        times = []
        for _ in range(3):
            tick = perf_counter()
            result = predict(CASES[name]["case"], rule, table)
            times.append(perf_counter() - tick)
    return result, dict(seconds=float(np.median(times)), first=setup + times[0], setup=setup, repeats=times)


def run():
    folder = RESULTS / "verification"
    raw = folder / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(OPERATING):
        order = ("operator", "1", "2", "3")
        for method in order if i % 2 == 0 else reversed(order):
            path = raw / f"{name}_{method}"
            if path.with_suffix(".json").exists() and path.with_suffix(".npz").exists():
                continue
            if method == "operator":
                data, stats = time_operator(name, raw)
            else:
                # Exclude imports and process startup from measured solver work.
                from importlib import import_module

                import_module("fea.solver")
                start = perf_counter()
                data, dofs = calculate_reference(CASES[name]["case"], int(method))
                stats = dict(seconds=perf_counter() - start, dofs=dofs)
            np.savez_compressed(path.with_suffix(".npz"), **data)
            path.with_suffix(".json").write_text(json.dumps(stats))
    rows, timings = summarize()
    save_csv(folder / "refinement.csv", rows)
    save_csv(folder / "timing.csv", timings)
    direct_check()
