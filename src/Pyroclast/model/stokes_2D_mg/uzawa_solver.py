
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

from .multigrid import Multigrid
from .smoother import pressure_sweep
from .mg_routines import uzawa_velocity_rhs
from .implicit_operators import p_residual, vx_residual, vy_residual
from .anderson import AndersonAccelerator
from .utils import apply_BC
from .viscosity_rescaler import ViscosityRescaler
from .residual_tracker import ResidualTracker

class UzawaSolver:
    """Solve the Stokes system using Uzawa iterations and multigrid."""

    def __init__(self, ctx, levels, scaling=2.0, accel_m=30):
        """Initialize the solver and allocate working arrays."""
        self.ctx = ctx

        # Set up multigrid solver for the velocity field
        self.mg = Multigrid(ctx, levels, scaling)
        self.rescaler = ViscosityRescaler(ctx, self.mg.hierarchy)

        # Allocate space for pressure solution and residuals
        ny1, nx1 = self.fine.ny1, self.fine.nx1
        self.p = np.zeros((ny1, nx1), dtype=np.float64)
        self.p_res = np.zeros((ny1, nx1), dtype=np.float64)
        
        # Set up solver parameters
        self.relax_p = ctx.params.get("relax_p", 0.7)
        self.p_ref = ctx.params.get("p_ref", None)
        self.BC = ctx.params.BC

        # Temporary storage for Anderson acceleration
        self.state_k = np.zeros((3, self.fine.ny1, self.fine.nx1))
        self.state_next = np.zeros_like(self.state_k)
        
        self.accel = AndersonAccelerator(m=accel_m, shape=(self.fine.ny1, self.fine.nx1))
        # Set the residual tracker to use the same window size as the accelerator
        # We need to ensure that AA can build up a long enough history to be effective
        # before signaling convergence.
        self.tracker = ResidualTracker(m=accel_m, tol=1e-19, convergence_thresh=2.5e-2)

    @property
    def fine(self):
        """Return the finest grid level."""
        return self.mg.hierarchy[0]
    
    def compute_residuals(self, p_rhs, vx_rhs, vy_rhs):
        """Return residual arrays for pressure and velocity."""

        nx1, ny1 = self.fine.nx1, self.fine.ny1
        dx, dy = self.fine.dx, self.fine.dy
        vx, vy = self.fine.vx, self.fine.vy

        p_res = p_residual(nx1, ny1, dx, dy, vx, vy, self.p_res, p_rhs)

        vx_res = vx_residual(nx1, ny1, dx, dy,
                             self.fine.etap, self.fine.etab,
                             vx, vy, self.p, self.fine.vx_res, vx_rhs)

        vy_res = vy_residual(nx1, ny1, dx, dy,
                             self.fine.etap, self.fine.etab,
                             vx, vy, self.p, self.fine.vy_res, vy_rhs)

        return p_res, vx_res, vy_res

    def solve(self, p_rhs, vx_rhs, vy_rhs,
              p_guess=None, vx_guess=None, vy_guess=None,
              max_cycles=50, tol=1e-7,
              nu1=3, nu2=3, velocity_cycles=2):
        """Run Uzawa iterations until convergence."""

        if p_guess is not None:
            self.p[...] = p_guess
        if vx_guess is not None:
            self.fine.vx[...] = vx_guess
        if vy_guess is not None:
            self.fine.vy[...] = vy_guess

        # Compute viscosity contrast for residual normalization
        deta_norm = self.fine.viscosity_contrast()

        # Compute cell normalization factor for RMSE
        rmse_norm = np.sqrt(self.fine.nx1 * self.fine.ny1)

        for cycle in range(max_cycles):
            print(f"Cycle: {cycle}")

            # Update velocity right hand sides using the current pressure
            self.fine.vx_rhs, self.fine.vy_rhs = uzawa_velocity_rhs(self.fine.nx1, self.fine.ny1,
                                                                    self.fine.dx, self.fine.dy,
                                                                    vx_rhs, vy_rhs, self.p,
                                                                    self.fine.vx_rhs, self.fine.vy_rhs)

            # Save current state for Anderson acceleration
            self.state_k[0] = self.fine.vx
            self.state_k[1] = self.fine.vy
            self.state_k[2] = self.p

            # Multigrid velocity solve
            for _ in range(velocity_cycles):
                vx, vy = self.mg.vcycle(0, nu1, nu2)

            # Pressure sweep
            self.p = pressure_sweep(self.fine.nx1, self.fine.ny1,
                                    self.fine.dx, self.fine.dy,
                                    vx, vy, self.p,
                                    self.fine.etap,
                                    self.relax_p, p_rhs,
                                    p_ref=self.p_ref)

            # Anderson acceleration on (vx, vy, p)
            self.state_next[0] = vx
            self.state_next[1] = vy
            self.state_next[2] = self.p
            state_accel = self.accel.update(self.state_k, self.state_next)
            if state_accel is not None:
                self.fine.vx[:, :] = state_accel[0]
                self.fine.vy[:, :] = state_accel[1]
                self.p[:, :] = state_accel[2]

                dp = self.p_ref - self.p[1, 1]
                self.p += dp
                apply_BC(self.p, self.fine.vx, self.fine.vy, self.BC)

            # Compute residuals and their norms
            p_res, vx_res, vy_res = self.compute_residuals(p_rhs, vx_rhs, vy_rhs)
            p_res_rmse = np.linalg.norm(p_res) / rmse_norm
            vx_res_rmse = np.linalg.norm(vx_res) / (rmse_norm * deta_norm)
            vy_res_rmse = np.linalg.norm(vy_res) / (rmse_norm * deta_norm)

            print(f"RMSE residuals: p = {p_res_rmse:.2e}, "
                  f"vx = {vx_res_rmse:.2e}, vy = {vy_res_rmse:.2e}")

            # Update the residual tracker
            converged = self.tracker.update(p_res_rmse, vx_res_rmse, vy_res_rmse)
            if converged and self.rescaler.done_rescaling():
                    print("Converged!")
                    break

            if self.rescaler.update_viscosity(): # Return True if viscosity was rescaled
                # Reset the accelerator, tracker and viscosity contrast
                self.accel.reset()
                self.tracker.reset()
                deta_norm = self.fine.viscosity_contrast()

        return self.p, self.fine.vx, self.fine.vy

