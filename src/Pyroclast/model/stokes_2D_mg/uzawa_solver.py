
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
        self.mg = Multigrid(ctx, levels, scaling)
        self.fine = self.mg.hierarchy[0]
        self.rescaler = ViscosityRescaler(ctx, self.mg.hierarchy)

        self.relax_p = ctx.params.get("relax_p", 0.7)
        self.p_ref = ctx.params.get("p_ref", None)
        self.BC = ctx.params.BC

        # Temporary storage for Anderson acceleration
        self.state_k = np.zeros((3, self.fine.ny1, self.fine.nx1))
        self.state_next = np.zeros_like(self.state_k)
        self.accel = AndersonAccelerator(m=accel_m, shape=(self.fine.ny1, self.fine.nx1))

        self.tracker = ResidualTracker(m=30, tol=1e-16, convergence_thresh=1e-2)

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
              nu1=3, nu2=3, gamma=1, velocity_cycles=2):
        """Run Uzawa iterations until convergence."""

        fine = self.fine

        # Solution and residual arrays
        self.p = np.zeros((fine.ny1, fine.nx1), dtype=np.float64)
        self.p_res = np.zeros((fine.ny1, fine.nx1), dtype=np.float64)

        if p_guess is not None:
            self.p[...] = p_guess
        if vx_guess is not None:
            fine.vx[...] = vx_guess
        if vy_guess is not None:
            fine.vy[...] = vy_guess

        for cycle in range(max_cycles):
            print(f"Cycle: {cycle}")

            # Update velocity right hand sides using the current pressure
            fine.vx_rhs, fine.vy_rhs = uzawa_velocity_rhs(
                fine.nx1, fine.ny1,
                fine.dx, fine.dy,
                vx_rhs, vy_rhs, self.p,
                fine.vx_rhs, fine.vy_rhs)

            # Save current state for Anderson acceleration
            self.state_k[0] = fine.vx
            self.state_k[1] = fine.vy
            self.state_k[2] = self.p

            # Multigrid velocity solve
            for _ in range(velocity_cycles):
                vx, vy = self.mg.vcycle(0, nu1, nu2)

            # Pressure sweep
            self.p = pressure_sweep(fine.nx1, fine.ny1,
                                    fine.dx, fine.dy,
                                    vx, vy, self.p,
                                    fine.etap,
                                    self.relax_p, p_rhs,
                                    p_ref=self.p_ref)

            # Anderson acceleration on (vx, vy, p)
            self.state_next[0] = vx
            self.state_next[1] = vy
            self.state_next[2] = self.p
            state_accel = self.accel.update(self.state_k, self.state_next)
            if state_accel is not None:
                fine.vx[:, :] = state_accel[0]
                fine.vy[:, :] = state_accel[1]
                self.p[:, :] = state_accel[2]

                dp = self.p_ref - self.p[1, 1]
                self.p += dp
                apply_BC(self.p, fine.vx, fine.vy, self.BC)

            # Compute residuals and their norms
            p_res, vx_res, vy_res = self.compute_residuals(p_rhs, vx_rhs, vy_rhs)

            dEta = fine.viscosity_contrast()
            N = np.sqrt(fine.nx1 * fine.ny1)
            p_res_rmse = np.linalg.norm(p_res) / N
            vx_res_rmse = np.linalg.norm(vx_res) / (N * dEta)
            vy_res_rmse = np.linalg.norm(vy_res) / (N * dEta)

            print(
                f"RMSE residuals: p = {p_res_rmse:.2e}, "
                f"vx = {vx_res_rmse:.2e}, vy = {vy_res_rmse:.2e}")

            # Update the residual tracker
            converged = self.tracker.update(p_res_rmse, vx_res_rmse, vy_res_rmse)

            if max(p_res_rmse, vx_res_rmse, vy_res_rmse) < tol and \
               self.rescaler.done_rescaling():
                print("Converged!")
                break

            if self.rescaler.update_viscosity():
                self.accel.reset()
                self.tracker.reset()

        return self.p, fine.vx, fine.vy

