"""Independently isolate wall-table and log-time interpolation at off-anchor queries."""

from tempfile import TemporaryDirectory

import numpy as np
from scipy.interpolate import CubicSpline

from studies.convergence import BASE, CASES, CASES4, RULES, single, read_rule, save
from model.wall import WallTable


def main():
    z, weights = read_rule(RULES / "fitted.npz")
    anchors = np.unique(np.r_[single.TIMES, np.geomspace(0.0005, 100, 33)])
    assert len(anchors) == 37
    mid = np.sqrt(anchors[:-1] * anchors[1:])

    def supply(model, t, force_direct):
        return model.right + sum(
            (weight * model.frequency(node / t, force_direct)[0]).real for node, weight in zip(z, weights)
        )

    rows, anchor_values = [], []
    with TemporaryDirectory() as directory:
        for name in CASES4:
            case = CASES[name]["case"]
            model = single.BoundaryOperator(case, *BASE, 14)
            table_model = single.BoundaryOperator(case, *BASE, 14)
            exact_pe_model = single.BoundaryOperator(case, *BASE, 14)
            if case["Pe"]:
                table = WallTable(directory, 22, 28, 14)
                table.attach(table_model)
                coefficients = table.fit_node(case["Pe"])
                original_response = exact_pe_model.exterior.response

                def exact_pe_response(s, need_field=True):
                    if need_field:
                        return original_response(s, True)
                    value = (table.features(s) @ coefficients).reshape(table.eye.shape)
                    return value + np.sqrt(s) * table.eye, None

                exact_pe_model.exterior.response = exact_pe_response
            direct = [supply(model, t, False) for t in anchors]
            normal = [supply(table_model, t, t in single.TIMES) for t in anchors]
            at_flow = [supply(exact_pe_model, t, t in single.TIMES) for t in anchors]
            anchor_values.extend(
                dict(case=name, tau=t, direct_supply=d, production_supply=n, exact_pe_fit=e)
                for t, d, n, e in zip(anchors, direct, normal, at_flow)
            )
            direct_spline = CubicSpline(np.log(anchors), direct)(np.log(mid))
            normal_spline = CubicSpline(np.log(anchors), normal)(np.log(mid))
            exact_pe_spline = CubicSpline(np.log(anchors), at_flow)(np.log(mid))
            for t, ds, ns, es in zip(mid, direct_spline, normal_spline, exact_pe_spline):
                exact = supply(model, t, False)
                rows.append(
                    dict(
                        case=name,
                        tau=t,
                        direct_supply=exact,
                        time_spline_minus_direct=ds - exact,
                        table_effect_on_spline=ns - ds,
                        pe_interpolation_effect=ns - es,
                        frequency_fit_at_exact_pe=es - ds,
                        production_minus_direct=ns - exact,
                    )
                )
    save("interpolation.csv", rows)
    save("fluid_anchors.csv", anchor_values)


if __name__ == "__main__":
    main()
