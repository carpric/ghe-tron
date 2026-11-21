# Low-Péclet asymptotic benchmark

This comparison uses the 90° orientation of the asymmetric borehole in Rico and Hermanns, *International Journal of Thermal Sciences* 204 (2024), 109222. The 76 mm borehole radius is used to normalize the 20 and 25 mm pipe radii. The offsets are 51 and 46 mm and the bearings are 135° and 45° counterclockwise from downstream. The imposed leg powers are in a 5:1 ratio. The reported mean uses the same two weights for GHE-TRON, FE and the asymptotic expansion at each Pe. [Original article](https://doi.org/10.1016/j.ijthermalsci.2024.109222)

From the repository root:

```bash
python reproduce.py asymptotic
```

This rebuilds the two-panel comparison figure and integrates the relative errors on the 91 transient samples at each Pe. The asymptotic values in `data/` were calculated from the published equations. The release supplies those numerical series and the mean-temperature weights. The separate implementation that generated them is private.

To generate GHE-TRON and independent FE temperatures anew, use the FE environment:

```bash
python -m optional.asymptotic.compare --recompute
```

This calculation also refines the exterior radial discretization to resolve the low-frequency limit. For the FE sensitivity study, run `python -m optional.asymptotic.reference transient 2` or `python -m optional.asymptotic.reference steady 2`; levels 1–3 are supported. Outputs go to `results/asymptotic/`. At Pe = 0.10 and 0.50 the recorded time-weighted errors against FE are respectively 0.0674% and 0.0168% for GHE-TRON, and 0.367% and 5.782% for the first-order expansion.
