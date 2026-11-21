# Low-Péclet asymptotic benchmark

The benchmark uses the asymmetric borehole in Section 7 of J. Rico and M. Hermanns, “On the network of thermal resistances embodying the response of geothermal boreholes to creeping groundwater flows”, *International Journal of Thermal Sciences* 204 (2024), 109222. The source of the method is the [original article](https://doi.org/10.1016/j.ijthermalsci.2024.109222). This repository supplies an independent implementation in [expansion.py](expansion.py).

The borehole radius is 76 mm. The two pipe radii are 20 and 25 mm, their centre offsets are 51 and 46 mm, and their bearings after the 90° rotation are 135° and 45° counterclockwise from downstream. The grout and ground conductivities are 1.50 and 3.00 W/(m K). The pipe resistances in the source convention are 0.535 and 0.544 m K/W. The imposed leg powers per unit length are 5q_c and q_c, and the temperature scale is q_c divided by ground conductivity.

The solver expands each pipe field in 20 circular multipole orders, enforces its Robin condition by Fourier projection at 512 points per circle, and uses 64 harmonics for the first-order wall forcing (Appendix A of the article). Equations (23) and (24) give the transient outer temperature and gradient. Their steady limit gives each pipe temperature at positive Pe. At each Pe, the model computes the mean-temperature weights from the inverse resistance matrix. The same two weights define the observable for all three methods.

## Reproduce the comparison

Run from the repository root in the inference environment.

```bash
python reproduce.py asymptotic
```

This solves the expansion at the archived times and Péclet numbers. It checks the two mean-temperature weights per flow, the 91 transient samples at each of Pe = 0.1 and 0.5, and both steady pipe temperatures at 128 Péclet numbers from 0.01 to 1, including the ten flows used for the steady FE comparison. The check reports the maximum absolute difference and fails above `1e-11`. This tolerance measures reproduction of the archived calculation. The errors against FE measure the accuracy of the asymptotic approximation.

The figure uses the freshly computed expansion and the archived GHE-TRON and FE temperatures. The plotted values, mean-temperature weights, integrated errors and PDF are written under `results/asymptotic/`. To evaluate the expansion at other positive times or at steady state without loading benchmark data:

```python
import numpy as np
from optional.asymptotic.expansion import inner_problem

model = inner_problem(rotation=90.0)
pipe_temperatures = model.temperatures(0.1, [10.0, 100.0])
weighted_mean = pipe_temperatures @ model.weights(0.1)
steady_pipe_temperatures = model.temperatures(0.1, np.inf)
```

To generate GHE-TRON and independent FE temperatures anew, use the FE environment:

```bash
python -m optional.asymptotic.compare --recompute
```

This calculation also refines the exterior radial discretization to resolve the low-frequency limit. For the FE sensitivity study, run `python -m optional.asymptotic.reference transient 2` or `python -m optional.asymptotic.reference steady 2`; levels 1–3 are supported. Outputs go to `results/asymptotic/`. At Pe = 0.10 and 0.50 the recorded time-weighted errors against FE are respectively 0.0674% and 0.0168% for GHE-TRON, and 0.367% and 5.782% for the first-order expansion.
