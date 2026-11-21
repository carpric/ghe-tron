"""Reproduce the heat-exchanger studies. Run from the repository root."""

import argparse
from shutil import copyfile
import numpy as np

from studies.inputs import CASES, DATA, RESULTS, ROOT, save_csv


def single_panel(names):
    from studies.unit import unit

    dest = RESULTS if not names else RESULTS / "subsets" / "_".join(names)
    rows, operating = unit(names or CASES)
    save_csv(dest / "casewise_FE.csv", rows)
    if operating:
        save_csv(dest / "operating_fine_grid.csv", operating)


def parallel_panel():
    from studies import parallel, square
    from fea.parallel import run

    parallel.RAW.mkdir(parents=True, exist_ok=True)
    primary, third = [], []
    for case in parallel.load_cases():
        parallel.predict(case)
        levels = [1, 2, 4] if case["id"] in {"D06", "D08", "T01"} else [1, 2]
        for level in levels:
            if not all(
                parallel.path(case, f"fe{level}", signed).exists()
                for signed in ([False, True] if case["signed"] else [False])
            ):
                parallel.reference((case, level))
        for signed in [False, True] if case["signed"] else [False]:
            primary.extend(parallel.scores(case, signed))
            if 4 in levels:
                third.extend(parallel.scores(case, signed, 4))
    save_csv(RESULTS / "parallel_assessment.csv", primary)
    save_csv(RESULTS / "parallel_third_level.csv", third)

    square.predict()
    for level in (1, 2):
        for pe in (0.0, 2.0, 5.0):
            run(pe, level)
        run(2.0, level, signed=True)
    square.measure()
    square.signed()
    for source, target in [
        ("metrics", "square_unit_step"),
        ("observations", "square_observations"),
        ("matched_contrasts", "square_contrasts"),
        ("signed_metrics", "square_signed"),
    ]:
        copyfile(square.HERE / (source + ".csv"), RESULTS / (target + ".csv"))
    histories = []
    for fraction in (0.5, 0.35):
        name = square.name("opposite", 2.0, fraction)
        with np.load(square.RAW / f"{name}_op.npz") as op, np.load(square.RAW / f"{name}_fe2.npz") as fe:
            for t, a, b in zip(op["time"], op["fluid"], fe["fluid"], strict=True):
                histories.append(
                    dict(
                        first_branch_fraction=fraction,
                        tau=t,
                        return_1_operator=a[1],
                        return_2_operator=a[2],
                        return_1_fine_FE=b[1],
                        return_2_fine_FE=b[2],
                    )
                )
    save_csv(RESULTS / "square_return_histories.csv", histories)


def verification():
    from studies import convergence, interpolation, physics, stress
    from studies.unit import temporal

    save_csv(RESULTS / "time_inversion.csv", temporal(CASES))
    convergence.curves()
    convergence.spatial()
    stress.main()
    interpolation.main()
    physics.run()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    tasks = ["summary", "figures", "single", "wake", "parallel", "groundwater", "verification", "timing", "quasi-steady", "conductive-resistance", "all"]
    if (ROOT / "optional/asymptotic/compare.py").is_file():
        tasks.append("asymptotic")
    p.add_argument("study", choices=tasks)
    p.add_argument("--case", nargs="+", choices=list(CASES), help="Subset for the single-U FE study")
    p.add_argument("--computed", action="store_true", help="Read results/ when summarizing or plotting")
    args = p.parse_args()
    if args.case and args.study != "single":
        p.error("--case applies to single")
    if args.computed and args.study not in ("summary", "figures"):
        p.error("--computed applies to summary and figures")
    RESULTS.mkdir(exist_ok=True)
    steps = (
        ("single", "wake", "parallel", "groundwater", "verification", "timing", "quasi-steady", "conductive-resistance", "summary", "figures")
        if args.study == "all"
        else (args.study,)
    )
    data = RESULTS if args.computed or args.study == "all" else DATA / "reported"
    for step in steps:
        print(f"Running {step}", flush=True)
        if step == "single":
            single_panel(args.case)
        elif step == "wake":
            from studies import wake

            wake.run()
        elif step == "parallel":
            parallel_panel()
        elif step == "verification":
            verification()
        elif step == "groundwater":
            from studies import groundwater

            groundwater.main()
        elif step == "timing":
            from studies import desktop_timing, interpolation

            if not (RESULTS / "verification/fluid_anchors.csv").exists():
                interpolation.main()
            desktop_timing.run()
            from studies.benchmark import run

            run()
        elif step == "quasi-steady":
            from studies.quasi_steady import run
            run()
        elif step == "conductive-resistance":
            from studies.conduction_resistance import run
            run()
        elif step == "asymptotic":
            from optional.asymptotic.compare import present

            present()
        else:
            from studies import plots

            getattr(plots, step)(data)


if __name__ == "__main__":
    main()
