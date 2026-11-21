"""Harmonic outer resistance for the two-pipe example of Rivero & Hermanns (2021).

Equation (15) defines an apparent ground temperature using only monopole terms.
Equation (18) uses the inverse matrix of fluid minus apparent temperatures,
obtained here from two independent unit pipe powers in W/m.
"""
import numpy as np
from scipy.special import iv, kv

from model.response import Circuit, MultiPipeResponse
from studies.inputs import RESULTS, save_csv


def resistance(beta_ground, order=12):
    kg, ks = 1.8, 1.4
    radii = np.array([20., 25.]) / 76.
    angles = np.deg2rad([0., 61.80])
    offsets = np.array([51., 46.]) / 76.
    xy = offsets[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    s = 1j * beta_ground**2
    g = np.sqrt(s * (8.7 / 4.6))
    p = np.sqrt(s)
    ground_impedance = p * kv(1,p) / kv(0,p)
    incident_derivative = -g * kv(1,g) / kv(0,g)
    regular_derivative = g * iv(1,g) / iv(0,g)
    # Regular zero mode induced at the borehole by each singular K0 source.
    reflection = -(ground_impedance + kg/ks * incident_derivative) / (ground_impedance + kg/ks * regular_derivative)
    apparent_per_monopole = kv(0,g) * (1 + reflection * iv(0,g*offsets))
    columns = []
    for fractions in np.eye(2):
        model = MultiPipeResponse(xy, radii, kg*radii*np.array([.540,.551]), 2*np.pi*kg,
                                  kg/ks, 8.7/4.6, 0., 0., Circuit.prescribed_heat_fractions(fractions))
        model.n = np.arange(-order, order+1)
        transfer, state = model.frequency(s, True)
        monopoles = state[2][:,order] / kv(0,g*radii)
        apparent = apparent_per_monopole @ monopoles
        columns.append(model.right + transfer - apparent)
    matrix = np.column_stack(columns)
    conductances = np.linalg.solve(matrix.T, np.ones(2))
    return 1 / conductances.sum()


def run():
    rows = []
    for beta in (.001,.01,1.,3.):
        value = resistance(beta)
        refined = resistance(beta, 20)
        rows.append(dict(beta_ground_abs=beta, resistance_abs_mK_per_W=abs(value),
                         phase_rad=np.angle(value), order_refinement_absolute=abs(value-refined)))
    save_csv(RESULTS / "quasi_steady" / "conductive_resistance.csv", rows)
    return rows


if __name__ == "__main__":
    for row in run():
        print(row)
