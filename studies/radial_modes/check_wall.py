"""Reproduce the 24 scalar wall-impedance comparisons in the supplied experiment."""
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .exact_mathieu import angular_eigs, collocation_Z, exact_Z, shooting_Z

CASES = ((0.5, 1.), (0.5, .1+.3j), (1.5, .05), (1.5, 7+3j),
         (4.5, .07+.03j), (4.5, 20+10j))


def compare(case):
    k, s, index = case
    a = angular_eigs(k)[index]
    z, nu, residual = exact_Z(a, k, s, J=40)
    return dict(Pe=2*k, s_real=s.real, s_imag=s.imag, mode_index=index,
                a=a, nu_real=nu.real, nu_imag=nu.imag, Z_real=z.real, Z_imag=z.imag,
                abs_diff_shoot=abs(z-shooting_Z(a,k,s)),
                abs_diff_29=abs(z-collocation_Z(a,k,s,28)),
                abs_diff_201=abs(z-collocation_Z(a,k,s,200)), floquet_residual=residual)


if __name__ == "__main__":
    inputs = [(k, s, i) for k, s in CASES for i in range(4)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(compare, inputs))
    output = Path("results/radial_modes")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "wall.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for col in ("abs_diff_shoot", "abs_diff_29", "abs_diff_201"):
        print(col, max(row[col] for row in rows))
