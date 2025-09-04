
"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: uzawa_solver.py
Description: This file implements the Uzawa solver for the Stokes flow
              and continuity equations in 2D.
                    
Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np

# Use the solver-agnostic multigrid base with Stokes-specific hooks
from Pyroclast.logging import get_logger
from Pyroclast.context import ContextNamespace
from Pyroclast.solvers.stokes_2d.bc import apply_BC

from .grid_hierarchy import GridHierarchy
from .velocity_multigrid_2D import VelocityMultigrid2D
from .smoother import pressure_sweep
from .implicit_operators import uzawa_velocity_rhs, \
                                p_residual, uzawa_vx_residual, uzawa_vy_residual, \
                                compute_p_energy_norm, compute_vx_energy_norm, compute_vy_energy_norm
from .viscosity_rescaler import ViscosityRescaler


logger = get_logger(__name__)

class UzawaSolver:
    """Solve the Stokes system using Uzawa iterations and multigrid."""

    def __init__(self, ctx, nlevels, scaling=2.0):
        """Initialize the solver and allocate working arrays."""
        s, p, o = ctx
        
        # Set up multigrid solver for the velocity field
        self.hierarchy = GridHierarchy(ctx, nlevels, scaling)
        self.mg = VelocityMultigrid2D(self.hierarchy, scaling)
        self.fine = self.hierarchy[0] # Fine grid solution
        
        # Read Resolution from fine grid
        # The pressure sub-problem should match this
        self.nx1 = self.fine.nx1
        self.ny1 = self.fine.ny1
        self.dx = self.fine.dx
        self.dy = self.fine.dy

        # For convenience, we extract the fine
        # grid solution from the velocity mg solver
        self.vx = self.fine.vx
        self.vx_res = self.fine.vx_res
        self.vx_rhs = self.fine.vx_rhs
        self.vy = self.fine.vy
        self.vy_res = self.fine.vy_res
        self.vy_rhs = self.fine.vy_rhs

        # Read material values from fine grid
        # This is important as we might be doing viscosity rescaling!
        self.etab = self.fine.etab
        self.etap = self.fine.etap
        self.rho = self.fine.rho

        # Store original stokes problem material properties
        self.stokes_etab = s.etab
        self.stokes_etap = s.etap
        self.stokes_rho = s.rho

        # Allocate memory for pressure solution and residual
        self.p = np.zeros((self.ny1, self.nx1))
        self.p_rhs = np.zeros((self.ny1, self.nx1))
        self.p_res = np.zeros((self.ny1, self.nx1))

        # Set up viscosity rescaler
        self.rescaler = ViscosityRescaler(ctx, self.hierarchy)

        # Set up solver parameters
        self.relax_p = p.get("relax_p", 0.7)
        self.relax_v = p.get("relax_v", 0.7)
        self.velocity_cycles = p.get("uzawa_velocity_cycles", 2)
        self.BC = p.BC  # type depends on your BC convention

    def compute_residuals(self):
        """Return residual arrays for pressure and velocity."""
        
        # Compute energy norm of residuals
        # Pressure residual
        self.p_res = p_residual(self.nx1, self.ny1,
                                self.dx, self.dy,
                                self.vx, self.vy,
                                self.p_res, self.stokes_p_rhs)
    
        # Velocity residuals
        self.vx_res = uzawa_vx_residual(self.nx1, self.ny1,
                                        self.dx, self.dy,
                                        self.stokes_etap, self.stokes_etab,
                                        self.vx, self.vy,
                                        self.vx_res, self.vx_rhs)

        self.vy_res = uzawa_vy_residual(self.nx1, self.ny1,
                                        self.dx, self.dy,
                                        self.stokes_etap, self.stokes_etab,
                                        self.vx, self.vy,
                                        self.vy_res, self.vy_rhs)

        # Compute energy norm residuals
        p_energy = compute_p_energy_norm(self.nx1, self.ny1, self.p_res, self.etap)
        vx_energy = compute_vx_energy_norm(self.nx1, self.ny1, self.dx, self.dy,
                                           self.vx_res, self.vx_rhs,
                                           self.stokes_etap, self.stokes_etab, normalize=False)
        vy_energy = compute_vy_energy_norm(self.nx1, self.ny1, self.dx, self.dy,
                                           self.vy_res, self.vy_rhs,
                                           self.stokes_etap, self.stokes_etab, normalize=False)

        return p_energy, vx_energy, vy_energy

    def solve(self, stokes_p_rhs, stokes_vx_rhs, stokes_vy_rhs,
              p_guess=None, vx_guess=None, vy_guess=None,
              max_cycles=1000, nu1=3, nu2=3, velocity_cycles=2):
        """Run Uzawa iterations until convergence."""

        # Set up initial guess if given
        if p_guess is not None:
            self.p[...] = p_guess
        if vx_guess is not None:
            self.vx[...] = vx_guess
        if vy_guess is not None:
            self.vy[...] = vy_guess

        # Store rhs for original stokes problem
        self.stokes_p_rhs = stokes_p_rhs
        self.stokes_vx_rhs = stokes_vx_rhs
        self.stokes_vy_rhs = stokes_vy_rhs

        for cycle in range(max_cycles):
            logger.debug(f"Cycle: {cycle}")
            # Perform Uzawa Sweep

            # Update velocity right hand sides using the current pressure
            # This routine computes the uzawa rhs from the original stokes problem
            # Pressure here is assumed to be constant
            self.vx_rhs, self.vy_rhs = uzawa_velocity_rhs(self.nx1, self.ny1,
                                                            self.dx, self.dy,
                                                            self.stokes_vx_rhs, self.stokes_vy_rhs, self.p,
                                                            self.vx_rhs, self.vy_rhs)

            # Multigrid solve for velocity
            # This updates vx and vy in-place
            for _ in range(self.velocity_cycles):
                self.mg.vcycle(0, nu1, nu2)

            # Pressure update
            self.p = pressure_sweep(self.nx1, self.ny1,
                                    self.dx, self.dy,
                                    self.vx, self.vy,
                                    self.p, 
                                    self.etap,
                                    self.relax_p, stokes_p_rhs)
            
            # # Reapply boundary conditions        
            # apply_BC(self.p, self.fine.vx, self.fine.vy, self.BC)

            # Compute residuals and their norms
            p_res, vx_res, vy_res = self.compute_residuals()

            logger.debug(
                f"RMSE residuals: p = {p_res:.2e}, "
                f"vx = {vx_res:.2e}, vy = {vy_res:.2e}")

            # Update viscosity
            self.rescaler.update_viscosity()

        return self.p, self.vx, self.vy

