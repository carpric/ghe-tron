# GHE-TRON

Research code for *A semi-analytical transient multipole method for fluid and ground temperatures of ground heat exchangers under groundwater flow*.

Given pipe positions and radii, fluid-circuit connections, material properties and groundwater flow, GHE-TRON computes unit-step fluid temperatures and grout and ground fields. At each complex frequency, a transient multipole solution in the grout couples the circuit and pipe walls to a temperature-to-flux map for the unbounded ground. A ten-node Laplace inversion rule, optimized once on scalar decaying modes and reused across exchangers, recovers the time response. Independent finite-element (FE) solutions assess the resulting temperatures.

## Run a prediction

Use Python 3.11 or later. From this directory, on Linux, macOS or Windows with WSL:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python infer.py Q221 --times 0 0.02 1 10 100 --field 10 3 0
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`. Inference runs on CPU. The first call for a flowing layout builds a reusable wall table under `results/wall_tables/`.

The command writes `results/prediction/fluids.csv` (supply, individual returns and flow-weighted mixed return) and, for the specified point, `fields.csv`. Times are dimensionless, coordinates are in borehole-radius units, and temperatures are rises divided by the temperature scale. The supported time interval is 0–100. At zero, the unit-step fluid output gives the algebraic right limit and the solid temperature rise is zero.

To predict a signed load on an asymmetric double-U exchanger:

```bash
python infer.py examples/double_u.json --load examples/load.csv --output results/double_u
```

The load CSV has columns `tau,q` on a uniform grid. The first row is the zero pre-load state. Each subsequent load is imposed over the interval ending at that row. The fluid histories are reconstructed from the step response by causal FFT convolution. A `--field` query always requests the **unit-step** solid field.

### Supply an exchanger

Use the example JSON as a template. `centres` and `radii` describe circular pipes inside the unit borehole. `pairs` lists zero-based supply and return leg indices. The positive branch mass-flow `fractions` sum to one. The remaining inputs follow the manuscript notation:

| Input | Meaning |
|---|---|
| `lam`, `gamma` | Grout-to-ground conductivity ratio and ground-to-grout diffusivity ratio |
| `beta`, `coupling` | Per-leg wall coefficients and common coupling parameter C, with ψₖ = C Rₚₖ / βₖ |
| `pe`, `flow_angle_deg` | Groundwater Péclet number and direction counterclockwise from the positive x-axis. The wall table supports 0 ≤ Pe ≤ 10. |

The circuit closure gives one representative fluid temperature per leg. The assessed circuits are single-, double- and triple-U exchangers. For a general linear circuit use `Circuit(A, B, load_vector)` with `MultiPipeResponse` in `model/response.py`. The general API exposes individual fluid temperatures and both solid fields.

## Reproduce the reported results

The following commands recreate the summary statistics and numerical plots from the archived measurements and check the public solver against saved responses. They require only the inference environment.

```bash
python reproduce.py summary
python reproduce.py figures
python -m unittest discover -s tests
```

For example, the 24 single-U cases give median full-history inlet and outlet discrepancies of **0.02709% and 0.04635%** against fine FE. The median downstream field discrepancy at τ = 100 is **0.04159%** across the 18 flowing cases. The summary reports case counts and paired coarse–fine differences alongside the errors. Full casewise results and verification data are in `data/reported/`. The plots include unit and signed histories, local and downstream fields, the dimensional groundwater example, branch returns, inversion order and error versus runtime.

To recompute the results from the equations, create the reference-solver environment. This includes DOLFINx, PETSc and Gmsh. Run one operating case before starting the panel:

```bash
conda env create -f environment-fea.yml
conda activate ghe-tron-fea
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python reproduce.py single --case Q222
python reproduce.py all
```

The complete study runs serially and includes third-level FE meshes. For a smaller run, the following commands can be invoked separately:

| Command after `python reproduce.py` | Computation |
|---|---|
| `single` | 24 single-U cases, two FE levels, four signed operating histories |
| `wake` | Strong-flow wake stationarity on six layouts and FE outer-radius sensitivity on Q421 |
| `parallel` | Square double-U circuit and flow-split comparisons, eight asymmetric double-U cases, one triple-U case, and third-level checks on D06, D08 and T01 |
| `groundwater` | Q221 under four Darcy fluxes in dimensional units |
| `verification` | Fixed and fitted rules against refined Euler inversion, modified Talbot comparison at orders 4–12, spatial and wall-table resolution, rotation and energy balance |
| `timing` | Three FE levels on four single-U cases, conditional Richardson estimates, serial operator timings and FFT throughput |
| `summary --computed` or `figures --computed` | Read fresh outputs from `results/`. Run the corresponding studies first. |

A subset writes tables under `results/subsets/`. The full panels write to `results/`; `data/` holds the original inputs and results. Machine timings and the independently remeshed FE differences can vary between installations. For a fresh timing measurement, use an empty `results/verification/raw/` directory.

### Check wake stationarity and FE outer radius

In the FE environment, run `python reproduce.py wake`. It computes any missing unit-step responses, reuses those already computed by `single`, and writes two tables under `results/`. The same checks run as part of `all`.

- [wake_stationarity.csv](data/reported/wake_stationarity.csv) compares the downstream temperature field at dimensionless times 20 and 100 for each strong-flow layout. The percentage change is normalized by the field norm at time 100 and is evaluated separately for the multipole solution and both FE meshes.
- [verification_radius.csv](data/reported/verification_radius.csv) compares fine FE solutions for Q421 with outer radii of 50 and 75 borehole radii. It reports the time-weighted supply and return discrepancies through time 100 and the downstream field discrepancies at times 10, 20 and 100. Each percentage is relative to the radius-75 solution. The outer radius applies only to the FE reference.

## Refit the frequencies and residues

The fitted and fixed-pole rules used in the assessment are supplied under `data/rules/`. To reproduce the fit itself, install PyTorch and run:

```bash
python -m pip install torch
python fit.py --device cpu
python infer.py Q221 --rule results/fit/fitted.npz
```

The fit uses 800 Adam updates and reports the scalar residual on independent evaluation points. `--count J` fits a different order (4–12). Coefficients from a refit may differ numerically across linear algebra implementations. All reported comparisons use the archived coefficients.

## Optional low-Péclet asymptotic comparison

For the manuscript with the low-Péclet comparison, run `python reproduce.py asymptotic`. This computes the curves from an independent implementation of the Rico and Hermanns expansion, checks them against the archived values, and rebuilds the figure using the GHE-TRON and FE results. The [optional benchmark](optional/asymptotic/README.md) documents the equations, inputs and command for recomputing all three methods.

## License and citation

The code is available under the [MIT License](LICENSE). Keep the copyright and permission notices with redistributed copies or substantial portions of the code.

If this software contributes to a publication, please cite the manuscript named above and the software version used. GitHub's **Cite this repository** feature reads the software citation from [CITATION.cff](CITATION.cff).
