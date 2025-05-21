"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: multigrid.py
Description: This file implements the multigrid method for the Stokes flow
             and continuity equations in 2D.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
import numpy as np
from Pyroclast.profiling import timer
from .grid_hierarchy import GridHierarchy
from .smoother import pressure_sweep
from .mg_routines import uzawa_velocity_rhs
from .implicit_operators import p_residual, vx_residual, vy_residual
from .anderson import AndersonAccelerator
from .utils import apply_BC


class Multigrid:
    """
    High-level solver coordinating V-cycles over a GridHierarchy.
    """
    def __init__(self, ctx, levels, scaling=2.0):
        self.scaling = scaling
        self.hierarchy = GridHierarchy(ctx, levels, scaling)
        self.p_ref = ctx.params.get('p_ref', None)
        self.relax_p = ctx.params.get('relax_p', 0.7)
        self.BC = ctx.params.BC

    def vcycle(self, level, nu1, nu2, gamma):
        # Extract the current grid level
        fine = self.hierarchy[level]

        # Pre-smoothing
        if nu1 > 0:
            fine.smooth(nu1)
        
        # Coarse correction
        if level + 1 < len(self.hierarchy):
            coarse = self.hierarchy[level + 1]
            
            # Compte residuals on fine grid
            fine.update_residual()
            
            # Restrict residuals to coarse grid
            coarse.restrict_residuals(fine)

            # Solve problem on coarse grid
            # for _ in range(gamma):
            self.vcycle(level + 1, nu1*self.scaling, nu2*self.scaling, gamma)
            
            # Prolongate correction to fine grid
            coarse.prolong_correction(fine)
            
            fine.apply_bc()
            
            coarse.reset_solution()
        
        # Post-smoothing and residual update
        if nu2 > 0:
            fine.smooth(nu2)
        
        # Update residuals
        fine.update_residual()
        
        return fine.vx, fine.vy

    def set_rhs(self, vx_rhs, vy_rhs, grid):
        grid.vx_rhs[...] = vx_rhs
        grid.vy_rhs[...] = vy_rhs

    def set_guess(self, p_guess, vx_guess, vy_guess, grid):
        # Set the initial guess for the fine grid
        self.p[...] = p_guess if p_guess is not None else 0.0
        grid.vx[...] = vx_guess if vx_guess is not None else 0.0
        grid.vy[...] = vy_guess if vy_guess is not None else 0.0

    def residuals(self, fine, p_rhs, vx_rhs, vy_rhs):
        # Compute residuals
        p_res = p_residual(fine.nx1, fine.ny1,
                           fine.dx, fine.dy,
                           fine.vx, fine.vy,
                           self.p_res, p_rhs)
        
        vx_res = vx_residual(fine.nx1, fine.ny1,
                             fine.dx, fine.dy,
                             fine.etap, fine.etab,
                             fine.vx, fine.vy, self.p, fine.vx_res, vx_rhs)
        
        vy_res = vy_residual(fine.nx1, fine.ny1,
                             fine.dx, fine.dy,
                             fine.etap, fine.etab,
                             fine.vx, fine.vy, self.p, fine.vy_res, vy_rhs)
        
        return p_res, vx_res, vy_res

    def solve(self, p_rhs, vx_rhs, vy_rhs,
              p_guess, vx_guess, vy_guess,
              max_cycles=50, tol=1e-7,
              nu1=3, nu2=3, gamma=1,
              p_ref=None):
        
        # Extract the fine grid
        fine = self.hierarchy[0]
        
        # Allocate memory for pressure solution
        self.p = np.zeros((fine.ny1, fine.nx1), dtype=np.float64)
        self.p_res = np.zeros((fine.ny1, fine.nx1), dtype=np.float64)

        # Set the initial guess for the fine grid
        self.set_guess(p_guess, vx_guess, vy_guess, fine)

        # Anderson acceleration
        acc = AndersonAccelerator(m=15, shape=(fine.ny1, fine.nx1))

        # Main Uzawa loop
        for cycle in range(max_cycles):
            print(f"Cycle: {cycle}")
            
            # 1. Construct Uzawa rhs
            fine.vx_rhs, fine.vy_rhs = uzawa_velocity_rhs(fine.nx1, fine.ny1,
                                                        fine.dx, fine.dy,
                                                        vx_rhs, vy_rhs, self.p,
                                                        fine.vx_rhs, fine.vy_rhs)

            # Store current full state before updates
            state_k = np.stack([fine.vx.copy(), fine.vy.copy(), self.p.copy()])

            # MG solve for velocity
            for _ in range(3):
                vx, vy = self.vcycle(0, nu1, nu2, gamma)

            # Update pressure (in-place)
            self.p = pressure_sweep(fine.nx1, fine.ny1, fine.dx, fine.dy,
                                    vx, vy, self.p,
                                    fine.etap,
                                    self.relax_p, p_rhs,
                                    p_ref=self.p_ref)

            # Construct new state after update
            state_next = np.stack([vx, vy, self.p.copy()])

            # Anderson acceleration on full state
            state_accel = acc.update(state_k, state_next)
            if state_accel is not None:
                fine.vx[:, :] = state_accel[0]
                fine.vy[:, :] = state_accel[1]
                self.p[:, :] = state_accel[2]

                # Fix boundary conditions
                apply_BC(self.p, fine.vx, fine.vy, self.BC)

            # Compute residuals
            p_res, vx_res, vy_res = self.residuals(fine, p_rhs, vx_rhs, vy_rhs)

            # 8. Compute the norm of the residuals
            N = np.sqrt(fine.nx1 * fine.ny1)
            p_res_rmse = np.linalg.norm(p_res) / N
            vx_res_rmse = np.linalg.norm(vx_res) / N
            vy_res_rmse = np.linalg.norm(vy_res) / N

            print(f"Continuity residual: {p_res_rmse}")
            print(f"X-momentum residual: {vx_res_rmse}")
            print(f"Y-momentum residual: {vy_res_rmse}")
                
            # Check convergence
            if max(p_res_rmse, vx_res_rmse, vy_res_rmse) < tol:
                break


        return self.p, fine.vx, fine.vy
