"""Check radial fields and normalization, including a steady odd-parity eigenmode."""
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh_tridiagonal
from scipy.special import kve

from .exact_mathieu import angular_eigs, floquet_coeffs, floquet_nu

ROOT = Path(__file__).resolve().parents[2]
RADII = np.array([1., 1.5, 2., 3., 5.])


def shooting(a, k, s):
    p = np.sqrt(complex(s) + k*k)
    start = np.log(1 + 60 / p.real)
    def potential(z):
        return a + (s + k*k)*np.exp(2*z) + k*k*np.exp(-2*z)
    def rhs(z, v):
        return [v[1], potential(z)*v[0]]
    sol = solve_ivp(rhs, (start, 0), np.array([1, -np.sqrt(potential(start))], complex),
                    method="DOP853", rtol=1e-12, atol=1e-14, t_eval=np.log(RADII[::-1]))
    if not sol.success:
        raise RuntimeError(sol.message)
    y, dy = sol.y[:, ::-1]
    return -dy[0]/y[0], y/y[0]


def evaluate(radii, k, s, nu, indices, coeffs, offset=0):
    p = mp.sqrt(mp.mpc(s)+k*k)
    values = []
    for r in radii:
        x, y = p*r, k/mp.mpf(r)
        orders = [nu+int(j)+offset for j in indices]
        holder = {v: mp.besselk(v,x) for v in orders}
        parts = [c*holder[v]*mp.besseli(abs(int(j)-offset),y) for j,c,v in zip(indices,coeffs,orders)]
        scale = mp.fsum(parts)
        if r == 1:
            if scale == 0:
                raise ArithmeticError("Vanishing wall normalization for this product offset")
            deriv = []
            for j,c,v in zip(indices,coeffs,orders):
                n = abs(int(j)-offset)
                # x K'_v = v K_v - x K_{v+1}; y I'_n = n I_n + y I_{n+1}.
                next_k = holder[v+1] if v+1 in holder else mp.besselk(v+1,x)
                deriv.append(c*((v*holder[v]-x*next_k)*mp.besseli(n,y)
                                -holder[v]*(n*mp.besseli(n,y)+y*mp.besseli(n+1,y))))
            z = -mp.fsum(deriv)/scale
            conditioning = mp.fsum(map(abs,parts))/abs(scale)
        values.append(scale)
    return complex(z), np.array([complex(v/values[0]) for v in values]), float(conditioning)


def check(case):
    label,k,s,idx = case
    a = angular_eigs(k)[idx]
    nu = floquet_nu(a,k*mp.sqrt(mp.mpc(s)+k*k))
    indices,c,_ = floquet_coeffs(a,k*mp.sqrt(mp.mpc(s)+k*k),nu,40)
    z, y, cancel = evaluate(RADII,k,s,nu,indices,c)
    ref_z, ref_y = shooting(a,k,s)
    return [dict(test=label, Pe=2*k, s_real=s.real, s_imag=s.imag, index=idx,
                 r=r, R_real=v.real, R_imag=v.imag, abs_profile_error=abs(v-ref),
                 abs_impedance_error=abs(z-ref_z), wall_cancellation=cancel)
            for r,v,ref in zip(RADII,y,ref_y)]


def limiting_checks():
    # At s=0, the even-harmonic sine band has nu=0 and c[-j]=-c[j].
    k, J = 1.5, 40
    j = np.arange(1,J+1)
    a, vec = eigh_tridiagonal((2.*j)**2, k*k*np.ones(J-1), select="i", select_range=(0,0))
    c = [mp.mpf(v) for v in vec[:,0]]
    anti = [-v for v in reversed(c)]+[mp.mpf(0)]+c
    idx = np.arange(-J,J+1)
    raw = mp.fsum(v*mp.besselk(int(n),k)*mp.besseli(abs(int(n)),k) for n,v in zip(idx,anti))
    z, prof, _ = evaluate(RADII,k,0.,0,idx,anti,offset=1)
    ref_z, ref_prof = shooting(a[0],k,0.)
    print("Steady sine band, offset 0 raw sum:", mp.nstr(raw), "offset 1 abs Z, profile errors:",
          abs(z-ref_z), max(abs(prof-ref_prof)))
    for m in (0,3,12):
        s = .1+.3j
        p = np.sqrt(s)
        z = p*kve(m+1,p)/kve(m,p)-m
        ref_z, _ = shooting(m*m,0.,s)
        print("Conductive limit, Bessel order",m,"abs Z vs shooting",abs(z-ref_z))


if __name__ == "__main__":
    with np.load(ROOT/"data/rules/fitted.npz") as archive:
        poles = archive["z"]
    cases = [("positive",.5,1.,0), ("complex",1.5,7+3j,3),
             ("near_steady",.5,1e-12,0), ("continued_negative",.5,-.1,0),
             ("rule_first_tau10",1.5,poles[0]/10.,12),
             ("rule_last_tau100",4.5,poles[-1]/100.,22)]
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = [r for batch in pool.map(check,cases) for r in batch]
    output = Path("results/radial_modes")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "profiles.csv").open("w", newline="") as out:
        writer=csv.DictWriter(out,rows[0],lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Max profile difference:",max(r["abs_profile_error"] for r in rows))
    limiting_checks()
