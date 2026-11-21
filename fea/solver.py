"""FEniCSx mesh assembly and transient heat transfer with groundwater flow."""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import fem, io
from dolfinx.fem import Function, functionspace, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
import ufl
from ufl import grad, inner
import gmsh
from typing import Callable, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class NonDimParams:
    """Non-dimensional parameters derived from dimensional ones."""

    lam: float  # λ = k_b / k_s_eff
    gamma: float  # γ = α_s_eff / α_b
    beta1: float  # β₁ = b₁ R₁ / R₀
    beta2: float  # β₂ = b₂ R₂ / R₀
    psi1: float  # ψ₁ = H / (E₁ ṁ cₚ)
    psi2: float  # ψ₂ = H / (E₂ ṁ cₚ)
    q0_hat: float  # q̂₀ = q_total H / (ṁ cₚ ΔT_ref)

    # Groundwater
    Pe: float  # Peclet Number = (ρ c_p)_f R0 u / k_s_eff
    phi_v: float  # Flow direction angle

    # Geometry (dimensionless)
    xp1_star: float
    yp1_star: float
    xp2_star: float
    yp2_star: float
    Rp1_star: float
    Rp2_star: float
    R_far_star: float


def create_ghe_mesh(
    R0: float = 1.0,
    R_far: float = 40.0,
    xp1: float = -0.4,
    yp1: float = 0.0,
    xp2: float = 0.4,
    yp2: float = 0.0,
    Rp1: float = 0.1,
    Rp2: float = 0.1,
    mesh_size_pipe: float = 0.02,
    mesh_size_borehole: float = 0.05,
    mesh_size_far: float = 1.0,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> Tuple[dolfinx.mesh.Mesh, dict]:
    """
    Create mesh for GHE cross-section using Gmsh.

    Returns:
        msh: DOLFINx mesh
        markers: Dictionary with cell and facet markers
    """
    # Only rank 0 creates the mesh, then it's distributed (within the provided communicator)
    rank = comm.rank

    gmsh.initialize()
    if rank == 0:
        gmsh.model.add("GHE")

        # Points
        # Borehole center
        gmsh.model.geo.addPoint(0, 0, 0, mesh_size_borehole, 1)

        # Borehole boundary points
        gmsh.model.geo.addPoint(R0, 0, 0, mesh_size_borehole, 2)
        gmsh.model.geo.addPoint(0, R0, 0, mesh_size_borehole, 3)
        gmsh.model.geo.addPoint(-R0, 0, 0, mesh_size_borehole, 4)
        gmsh.model.geo.addPoint(0, -R0, 0, mesh_size_borehole, 5)

        # Far-field boundary points
        gmsh.model.geo.addPoint(R_far, 0, 0, mesh_size_far, 6)
        gmsh.model.geo.addPoint(0, R_far, 0, mesh_size_far, 7)
        gmsh.model.geo.addPoint(-R_far, 0, 0, mesh_size_far, 8)
        gmsh.model.geo.addPoint(0, -R_far, 0, mesh_size_far, 9)

        # Pipe 1 center and boundary
        gmsh.model.geo.addPoint(xp1, yp1, 0, mesh_size_pipe, 10)
        gmsh.model.geo.addPoint(xp1 + Rp1, yp1, 0, mesh_size_pipe, 11)
        gmsh.model.geo.addPoint(xp1, yp1 + Rp1, 0, mesh_size_pipe, 12)
        gmsh.model.geo.addPoint(xp1 - Rp1, yp1, 0, mesh_size_pipe, 13)
        gmsh.model.geo.addPoint(xp1, yp1 - Rp1, 0, mesh_size_pipe, 14)

        # Pipe 2 center and boundary
        gmsh.model.geo.addPoint(xp2, yp2, 0, mesh_size_pipe, 15)
        gmsh.model.geo.addPoint(xp2 + Rp2, yp2, 0, mesh_size_pipe, 16)
        gmsh.model.geo.addPoint(xp2, yp2 + Rp2, 0, mesh_size_pipe, 17)
        gmsh.model.geo.addPoint(xp2 - Rp2, yp2, 0, mesh_size_pipe, 18)
        gmsh.model.geo.addPoint(xp2, yp2 - Rp2, 0, mesh_size_pipe, 19)

        # Circles
        # Borehole
        gmsh.model.geo.addCircleArc(2, 1, 3, 1)
        gmsh.model.geo.addCircleArc(3, 1, 4, 2)
        gmsh.model.geo.addCircleArc(4, 1, 5, 3)
        gmsh.model.geo.addCircleArc(5, 1, 2, 4)

        # Far-field
        gmsh.model.geo.addCircleArc(6, 1, 7, 5)
        gmsh.model.geo.addCircleArc(7, 1, 8, 6)
        gmsh.model.geo.addCircleArc(8, 1, 9, 7)
        gmsh.model.geo.addCircleArc(9, 1, 6, 8)

        # Pipe 1
        gmsh.model.geo.addCircleArc(11, 10, 12, 9)
        gmsh.model.geo.addCircleArc(12, 10, 13, 10)
        gmsh.model.geo.addCircleArc(13, 10, 14, 11)
        gmsh.model.geo.addCircleArc(14, 10, 11, 12)

        # Pipe 2
        gmsh.model.geo.addCircleArc(16, 15, 17, 13)
        gmsh.model.geo.addCircleArc(17, 15, 18, 14)
        gmsh.model.geo.addCircleArc(18, 15, 19, 15)
        gmsh.model.geo.addCircleArc(19, 15, 16, 16)

        # Curve loops
        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)  # Borehole
        gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 2)  # Far-field
        gmsh.model.geo.addCurveLoop([9, 10, 11, 12], 3)  # Pipe 1
        gmsh.model.geo.addCurveLoop([13, 14, 15, 16], 4)  # Pipe 2

        # Surfaces
        gmsh.model.geo.addPlaneSurface([1, 3, 4], 1)  # Grout (inside borehole, outside pipes)
        gmsh.model.geo.addPlaneSurface([2, 1], 2)  # Ground (outside borehole)

        gmsh.model.geo.synchronize()

        # Physical groups
        # Domains
        gmsh.model.addPhysicalGroup(2, [1], 1)  # Grout
        gmsh.model.setPhysicalName(2, 1, "Grout")
        gmsh.model.addPhysicalGroup(2, [2], 2)  # Ground
        gmsh.model.setPhysicalName(2, 2, "Ground")

        # Boundaries
        gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], 10)  # Borehole wall (interface)
        gmsh.model.setPhysicalName(1, 10, "Interface")
        gmsh.model.addPhysicalGroup(1, [5, 6, 7, 8], 11)  # Far-field
        gmsh.model.setPhysicalName(1, 11, "FarField")
        gmsh.model.addPhysicalGroup(1, [9, 10, 11, 12], 12)  # Pipe 1 wall
        gmsh.model.setPhysicalName(1, 12, "Pipe1")
        gmsh.model.addPhysicalGroup(1, [13, 14, 15, 16], 13)  # Pipe 2 wall
        gmsh.model.setPhysicalName(1, 13, "Pipe2")

        # Generate mesh
        gmsh.model.mesh.generate(2)

    # Convert to DOLFINx. The gmsh bridge moved in 0.10 (dolfinx.io.gmshio ->
    # dolfinx.io.gmsh) and now returns a MeshData object instead of a 3-tuple, so
    # support both rather than pinning the environment to one DOLFINx release.
    _gmsh_io = getattr(io, "gmshio", None) or getattr(io, "gmsh", None)
    if _gmsh_io is None:
        raise ImportError(
            "DOLFINx has no gmsh bridge (looked for dolfinx.io.gmshio and dolfinx.io.gmsh). "
            "Install DOLFINx with gmsh support, or check that the `gmsh` Python package is "
            "importable in this environment."
        )
    _mesh_data = _gmsh_io.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    if hasattr(_mesh_data, "mesh"):  # DOLFINx >= 0.10
        msh = _mesh_data.mesh
        cell_markers = _mesh_data.cell_tags
        facet_markers = _mesh_data.facet_tags
    else:  # DOLFINx <= 0.9
        msh, cell_markers, facet_markers = _mesh_data

    gmsh.finalize()

    markers = {
        "cells": cell_markers,
        "facets": facet_markers,
        "grout": 1,
        "ground": 2,
        "interface": 10,
        "far_field": 11,
        "pipe1": 12,
        "pipe2": 13,
    }

    return msh, markers


# ============================================================
# FEM SOLVER
# ============================================================
class GHEFEMSolver:
    """
    Finite Element solver for transient GHE heat transfer.

    Solves the coupled grout-ground heat conduction/advection problem with:
    - Robin BC on pipe walls
    - Interface conditions at borehole wall
    - Far-field BC
    """

    def __init__(self, nondim_params: NonDimParams, mesh_refinement: int = 1):
        self.nondim_params = nondim_params
        self.comm = MPI.COMM_SELF

        # Set mesh sizes based on refinement
        refinement_factors = {1: 1.0, 2: 0.5, 3: 0.25}
        rf = refinement_factors[mesh_refinement]

        self.msh, self.markers = create_ghe_mesh(
            R0=1.0,
            R_far=self.nondim_params.R_far_star,
            xp1=self.nondim_params.xp1_star,
            yp1=self.nondim_params.yp1_star,
            xp2=self.nondim_params.xp2_star,
            yp2=self.nondim_params.yp2_star,
            Rp1=self.nondim_params.Rp1_star,
            Rp2=self.nondim_params.Rp2_star,
            mesh_size_pipe=0.02 * rf,
            mesh_size_borehole=0.05 * rf,
            mesh_size_far=1.0 * rf,
            comm=self.comm,
        )

        # Function space
        self.V = functionspace(self.msh, ("CG", 1))

        # Trial and test functions
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        # Solution functions
        self.theta = Function(self.V)  # Current temperature
        self.theta_n = Function(self.V)  # Previous temperature

        # Store results
        self.times = []
        self.T_f_in_history = []
        self.T_f_out_history = []
        self.theta_bar_1_history = []  # Wall-averaged temp pipe 1
        self.theta_bar_2_history = []  # Wall-averaged temp pipe 2
        self.theta_bar_bh_history = []  # Wall-averaged temp borehole wall

    def _compute_wall_averaged_temp(self, pipe: int) -> float:
        """
        Compute wall-averaged temperature at a pipe.

        Args:
            pipe: 1 or 2

        Returns:
            Wall-averaged temperature
        """
        # Get pipe boundary marker
        marker = self.markers["pipe1"] if pipe == 1 else self.markers["pipe2"]

        # Create measure for pipe boundary
        ds_pipe = ufl.Measure(
            "ds", domain=self.msh, subdomain_data=self.markers["facets"], subdomain_id=marker
        )

        # Compute integral of temperature over boundary (with MPI reduction)
        integral_T_local = fem.assemble_scalar(fem.form(self.theta * ds_pipe))
        integral_T = self.msh.comm.allreduce(integral_T_local, op=MPI.SUM)

        # Compute boundary length (with MPI reduction)
        one = fem.Constant(self.msh, PETSc.ScalarType(1.0))
        length_local = fem.assemble_scalar(fem.form(one * ds_pipe))
        length = self.msh.comm.allreduce(length_local, op=MPI.SUM)

        return integral_T / length if length > 0 else 0.0

    def _compute_borehole_wall_temp(self) -> float:
        """
        Compute wall-averaged temperature at the borehole wall (interface).

        Returns:
            Wall-averaged temperature at borehole boundary
        """
        # The borehole wall is an INTERNAL boundary between grout and ground
        # Use dS (interior facet measure) with (+) or (-) restriction
        dS_borehole = ufl.Measure(
            "dS",
            domain=self.msh,
            subdomain_data=self.markers["facets"],
            subdomain_id=self.markers["interface"],
        )

        # For interior facets, we need to use restriction ('+' or '-')
        # Both sides should give the same temperature if continuous
        # Compute integral of temperature over boundary (with MPI reduction)
        integral_T_local = fem.assemble_scalar(fem.form(self.theta("+") * dS_borehole))
        integral_T = self.msh.comm.allreduce(integral_T_local, op=MPI.SUM)

        # Compute boundary length (with MPI reduction)
        one = fem.Constant(self.msh, PETSc.ScalarType(1.0))
        length_local = fem.assemble_scalar(fem.form(one("+") * dS_borehole))
        length = self.msh.comm.allreduce(length_local, op=MPI.SUM)

        return integral_T / length if length > 0 else 0.0

    def _compute_fluid_temperatures(
        self,
        q_current: float,
        theta_bar_1: float,
        theta_bar_2: float,
    ) -> Tuple[float, float]:
        """
        Compute fluid temperatures from wall temperatures and heat rate.

        Uses the energy balance equations:
        q̂₀,k = ψ_k (Θ_f,k - θ̄_k)
        q̂₀ = Θ_f,in - Θ_f,out

        Args:
            q_current: Current heat rate (dimensional or non-dim)
            theta_bar_1: Wall-averaged temp at pipe 1
            theta_bar_2: Wall-averaged temp at pipe 2

        Returns:
            (T_f_in, T_f_out) or (Θ_f,in, Θ_f,out)
        """
        psi1 = self.nondim_params.psi1
        psi2 = self.nondim_params.psi2

        # From the two equations:
        # Θ_f,out = Θ_f,in - q̂₀
        # q̂₀ = ψ₁(Θ_f,in - θ̄₁) + ψ₂(Θ_f,in - q̂₀ - θ̄₂)
        # q̂₀ = (ψ₁ + ψ₂)Θ_f,in - ψ₁θ̄₁ - ψ₂θ̄₂ - ψ₂q̂₀
        # q̂₀(1 + ψ₂) = (ψ₁ + ψ₂)Θ_f,in - ψ₁θ̄₁ - ψ₂θ̄₂
        # Θ_f,in = [q̂₀(1 + ψ₂) + ψ₁θ̄₁ + ψ₂θ̄₂] / (ψ₁ + ψ₂)

        Theta_f_in = (q_current * (1 + psi2) + psi1 * theta_bar_1 + psi2 * theta_bar_2) / (psi1 + psi2)
        Theta_f_out = Theta_f_in - q_current

        return Theta_f_in, Theta_f_out

    def solve_transient(
        self,
        t_end: float,
        dt: float,
        q_profile: Optional[Callable[[float], float]] = None,
        save_every: int = 1,
        snapshot_times: Optional[list] = None,
        time_grid: Optional[np.ndarray] = None,
    ) -> Dict:
        """Backward Euler with an implicit fluid-wall Schur complement on the supplied time grid."""
        if not np.isfinite(dt) or not np.isfinite(t_end) or dt <= 0 or t_end <= 0:
            raise ValueError("dt and t_end must be finite and positive")
        if time_grid is None:
            n_steps = round(t_end / dt)
            if not np.isclose(n_steps * dt, t_end, rtol=1e-10, atol=1e-12):
                raise ValueError("t_end must be an integer multiple of dt")
            times_grid = np.linspace(0.0, t_end, n_steps + 1)
        else:
            times_grid = np.asarray(time_grid, dtype=float)
            if (
                times_grid.ndim != 1
                or len(times_grid) < 2
                or not np.all(np.isfinite(times_grid))
                or times_grid[0] != 0.0
                or not np.isclose(times_grid[-1], t_end, rtol=1e-10, atol=1e-12)
                or not np.all(np.diff(times_grid) > 0)
            ):
                raise ValueError("time_grid must increase strictly from zero to t_end")
            n_steps = len(times_grid) - 1
        if save_every < 1 or int(save_every) != save_every:
            raise ValueError("save_every must be a positive integer")
        if snapshot_times is not None:
            for target in snapshot_times:
                if (
                    not np.isfinite(target)
                    or target < 0
                    or target > t_end
                    or not np.any(np.isclose(times_grid, target, rtol=1e-10, atol=1e-12))
                ):
                    raise ValueError("Snapshot times must be within the horizon and on the time grid")
            snapshot_times = sorted(set(snapshot_times))

        # Default to constant heat rate
        if q_profile is None:
            q0 = self.nondim_params.q0_hat
            q_profile = lambda t: q0

        # Time stepping parameters
        t = 0.0

        # Initial condition: θ = 0
        self.theta.x.array[:] = 0.0
        self.theta_n.x.array[:] = 0.0

        # Boundary measures
        ds_pipe1 = ufl.Measure(
            "ds", domain=self.msh, subdomain_data=self.markers["facets"], subdomain_id=self.markers["pipe1"]
        )
        ds_pipe2 = ufl.Measure(
            "ds", domain=self.msh, subdomain_data=self.markers["facets"], subdomain_id=self.markers["pipe2"]
        )

        # Subdomain measures for grout and ground
        dx_grout = ufl.Measure(
            "dx", domain=self.msh, subdomain_data=self.markers["cells"], subdomain_id=self.markers["grout"]
        )
        dx_ground = ufl.Measure(
            "dx", domain=self.msh, subdomain_data=self.markers["cells"], subdomain_id=self.markers["ground"]
        )

        # For Robin BC, we need fluid temperatures
        # These are computed from the energy balance

        # MPI rank for controlling output (only rank 0 prints)

        # Clear history
        self.times = [0.0]
        self.T_f_in_history = [0.0]
        self.T_f_out_history = [0.0]
        self.theta_bar_1_history = [0.0]
        self.theta_bar_2_history = [0.0]
        self.theta_bar_bh_history = [0.0]

        # Snapshot storage for temperature field evolution
        if snapshot_times is not None:
            snapshot_times_sorted = sorted(snapshot_times)
            snapshots = {}
            next_snapshot_idx = 0
            # Store t=0 snapshot (initial condition)
            if 0.0 in snapshot_times_sorted:
                snapshots[0.0] = self.theta.x.array.copy()
                next_snapshot_idx = snapshot_times_sorted.index(0.0) + 1
                if next_snapshot_idx >= len(snapshot_times_sorted):
                    next_snapshot_idx = len(snapshot_times_sorted)
        else:
            snapshot_times_sorted = []
            snapshots = {}
            next_snapshot_idx = 0

        # ------------------------------------------------------------
        # PERFORMANCE: assemble matrix once + reuse KSP (LU) across time
        # ------------------------------------------------------------
        # The bilinear form is time-invariant for constant dt and constant coefficients.
        # Fluid temperatures only affect the RHS (linear form), so we can reuse the factorization.
        A = None
        ksp = None
        bc_far = None
        a_form = None
        response = None
        response_denominator = None
        coupling_rhs = None
        max_discrete_residual = 0.0
        max_coupling_defect = 0.0
        matrix_dt = None

        for step in range(n_steps):
            t = float(times_grid[step + 1])
            dt = float(times_grid[step + 1] - times_grid[step])
            rebuild = A is None or not np.isclose(dt, matrix_dt, rtol=1e-9, atol=1e-14)
            q_current = q_profile(t)

            # Compute wall-averaged temperatures from previous solution
            theta_bar_1 = self._compute_wall_averaged_temp(1)
            theta_bar_2 = self._compute_wall_averaged_temp(2)

            # Compute fluid temperatures
            T_f_in, T_f_out = self._compute_fluid_temperatures(q_current, theta_bar_1, theta_bar_2)

            # Get material properties for each subdomain
            lam = self.nondim_params.lam
            gamma = self.nondim_params.gamma

            gamma_grout = lam * gamma
            gamma_ground = 1.0
            k_grout = lam
            k_ground = 1.0
            beta1 = self.nondim_params.beta1
            beta2 = self.nondim_params.beta2

            # Advection velocity
            Pe = self.nondim_params.Pe
            phi_v = self.nondim_params.phi_v

            # Weak form (backward Euler) with proper subdomain integration
            # Grout: γ_g/dt (θ - θⁿ) v + k_g ∇θ·∇v = 0
            # Ground: γ_s/dt (θ - θⁿ) v + k_s ∇θ·∇v + Advection = 0

            # Fluid temperature constants
            T_f1 = fem.Constant(self.msh, PETSc.ScalarType(T_f_in))
            T_f2 = fem.Constant(self.msh, PETSc.ScalarType(T_f_out))
            beta1_c = fem.Constant(self.msh, PETSc.ScalarType(beta1))
            beta2_c = fem.Constant(self.msh, PETSc.ScalarType(beta2))
            dt_c = fem.Constant(self.msh, PETSc.ScalarType(dt))
            gamma_grout_c = fem.Constant(self.msh, PETSc.ScalarType(gamma_grout))
            gamma_ground_c = fem.Constant(self.msh, PETSc.ScalarType(gamma_ground))
            k_grout_c = fem.Constant(self.msh, PETSc.ScalarType(k_grout))
            k_ground_c = fem.Constant(self.msh, PETSc.ScalarType(k_ground))

            # Build/assemble bilinear form once (cached factorization)
            if rebuild:
                # Bilinear form
                # Grout domain
                a_form = (gamma_grout_c / dt_c) * self.u * self.v * dx_grout
                a_form += k_grout_c * inner(grad(self.u), grad(self.v)) * dx_grout

                # Ground domain
                a_form += (gamma_ground_c / dt_c) * self.u * self.v * dx_ground
                a_form += k_ground_c * inner(grad(self.u), grad(self.v)) * dx_ground

                # Advection term in ground (time-invariant if Pe,phi_v are constant)
                x = ufl.SpatialCoordinate(self.msh)
                r2 = x[0] ** 2 + x[1] ** 2 + 1e-10
                x_rot = x[0] * np.cos(phi_v) + x[1] * np.sin(phi_v)
                y_rot = -x[0] * np.sin(phi_v) + x[1] * np.cos(phi_v)
                ux_loc = 1.0 - (x_rot**2 - y_rot**2) / (r2**2)
                uy_loc = -2.0 * x_rot * y_rot / (r2**2)
                ux_global = ux_loc * np.cos(phi_v) - uy_loc * np.sin(phi_v)
                uy_global = ux_loc * np.sin(phi_v) + uy_loc * np.cos(phi_v)
                velocity = ufl.as_vector([ux_global, uy_global]) * Pe
                a_form += inner(velocity, grad(self.u)) * self.v * dx_ground

            # Add Robin BC terms
            scale_bc = lam / 1.0  # k_grout / k_ground
            term_coeff1 = scale_bc / beta1_c
            term_coeff2 = scale_bc / beta2_c

            # Add Robin BC to the bilinear form (only when we are (re)assembling A)
            if rebuild:
                a_form += term_coeff1 * self.u * self.v * ds_pipe1
                a_form += term_coeff2 * self.u * self.v * ds_pipe2

                # Far-field BC: θ = 0 (Dirichlet) — time-invariant
                far_facets = self.markers["facets"].find(self.markers["far_field"])
                far_dofs = fem.locate_dofs_topological(self.V, 1, far_facets)
                bc_far = fem.dirichletbc(PETSc.ScalarType(0.0), far_dofs, self.V)

                # Assemble matrix and create solver
                if ksp is not None:
                    ksp.destroy()
                if A is not None:
                    A.destroy()
                matrix_dt = dt
                A = assemble_matrix(form(a_form), bcs=[bc_far])
                A.assemble()

                ksp = PETSc.KSP().create(self.msh.comm)
                ksp.setType("preonly")
                pc = ksp.getPC()
                pc.setType("lu")
                # Prefer fast sparse LU if available
                try:
                    pc.setFactorSolverType("mumps")
                except Exception:
                    pass
                ksp.setOperators(A)
                ksp.setUp()

                if coupling_rhs is not None:
                    coupling_rhs.destroy()
                coupling_rhs = assemble_vector(
                    form(term_coeff1 * self.v * ds_pipe1 + term_coeff2 * self.v * ds_pipe2)
                )
                coupling_rhs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
                set_bc(coupling_rhs, [bc_far])  # homogeneous far boundary
                w = Function(self.V)
                ksp.solve(coupling_rhs, w.x.petsc_vec)
                if ksp.getConvergedReason() <= 0:
                    raise RuntimeError("Unit boundary-response solve failed")
                w.x.scatter_forward()
                response = w.x.array.copy()
                saved = self.theta.x.array.copy()
                self.theta.x.array[:] = response
                w1 = self._compute_wall_averaged_temp(1)
                w2 = self._compute_wall_averaged_temp(2)
                self.theta.x.array[:] = saved
                nd = self.nondim_params
                response_denominator = 1 - (nd.psi1 * w1 + nd.psi2 * w2) / (nd.psi1 + nd.psi2)
                if not np.isfinite(response_denominator) or abs(response_denominator) < 1e-10:
                    raise RuntimeError("Singular/ill-conditioned fluid coupling Schur complement")

            # Linear form with proper subdomain integration (RHS changes each step)
            L_form = (gamma_grout_c / dt_c) * self.theta_n * self.v * dx_grout
            L_form += (gamma_ground_c / dt_c) * self.theta_n * self.v * dx_ground
            L_form += term_coeff1 * T_f1 * self.v * ds_pipe1
            L_form += term_coeff2 * T_f2 * self.v * ds_pipe2

            b = assemble_vector(form(L_form))
            # Apply BC lifting + set bcs (same bc_far)
            apply_lifting(b, [form(a_form)], bcs=[[bc_far]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc_far])

            # Solve using cached KSP (LU factorization)
            x = self.theta.x.petsc_vec
            ksp.solve(b, x)
            if ksp.getConvergedReason() <= 0:
                raise RuntimeError(f"Thermal solve failed at t={t:g}")
            x.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

            z1 = self._compute_wall_averaged_temp(1)
            z2 = self._compute_wall_averaged_temp(2)
            nd = self.nondim_params
            delta = (nd.psi1 * (z1 - theta_bar_1) + nd.psi2 * (z2 - theta_bar_2)) / (
                (nd.psi1 + nd.psi2) * response_denominator
            )
            self.theta.x.array[:] += response * delta
            self.theta.x.scatter_forward()
            theta_bar_1 = self._compute_wall_averaged_temp(1)
            theta_bar_2 = self._compute_wall_averaged_temp(2)
            corrected_in, corrected_out = self._compute_fluid_temperatures(
                q_current, theta_bar_1, theta_bar_2
            )
            max_coupling_defect = max(
                max_coupling_defect, abs(corrected_in - T_f_in - delta), abs(corrected_out - T_f_out - delta)
            )
            T_f_in, T_f_out = corrected_in, corrected_out
            residual = b.duplicate()
            A.mult(x, residual)
            residual.axpy(-1, b)
            residual.axpy(-delta, coupling_rhs)
            denom = max(b.norm() + abs(delta) * coupling_rhs.norm(), 1e-30)
            max_discrete_residual = max(max_discrete_residual, residual.norm() / denom)
            residual.destroy()
            b.destroy()

            # Update previous solution
            self.theta_n.x.array[:] = self.theta.x.array[:]

            # Compute borehole wall temperature from current solution
            theta_bar_bh = self._compute_borehole_wall_temp()

            # Check for snapshot capture
            if next_snapshot_idx < len(snapshot_times_sorted):
                target_tau = snapshot_times_sorted[next_snapshot_idx]
                if np.isclose(t, target_tau, rtol=1e-10, atol=1e-12):
                    snapshots[target_tau] = self.theta.x.array.copy()
                    next_snapshot_idx += 1

            # Save results
            if step % save_every == 0 or step == n_steps - 1:
                self.times.append(t)
                self.T_f_in_history.append(T_f_in)
                self.T_f_out_history.append(T_f_out)
                self.theta_bar_1_history.append(theta_bar_1)
                self.theta_bar_2_history.append(theta_bar_2)
                self.theta_bar_bh_history.append(theta_bar_bh)

        results = {
            "times": np.array(self.times),
            "T_f_in": np.array(self.T_f_in_history),
            "T_f_out": np.array(self.T_f_out_history),
            "theta_bar_1": np.array(self.theta_bar_1_history),
            "theta_bar_2": np.array(self.theta_bar_2_history),
            "theta_bar_bh": np.array(self.theta_bar_bh_history),
            "final_field": self.theta.x.array.copy(),
            "max_coupling_defect": max_coupling_defect,
            "max_discrete_residual": max_discrete_residual,
        }

        # Add snapshots if requested
        if snapshot_times is not None:
            results["snapshots"] = snapshots
            results["dof_coords"] = self.V.tabulate_dof_coordinates()[:, :2].copy()

        if coupling_rhs is not None:
            coupling_rhs.destroy()
        if ksp is not None:
            ksp.destroy()
        if A is not None:
            A.destroy()
        return results
