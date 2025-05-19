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

from Pyroclast.profiling import timer
from .grid_hierarchy import GridHierarchy


class Multigrid:
    """
    High-level solver coordinating V-cycles over a GridHierarchy.
    """
    def __init__(self, ctx, levels, scaling=2.0):
        self.hierarchy = GridHierarchy(ctx, levels, scaling)

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
            for _ in range(gamma):
                self.vcycle(level + 1, nu1 * 2, nu2 * 2, gamma)
            
            # Prolongate correction to fine grid
            coarse.prolong_correction(fine)
            
            # Apply correction to fine grid
            # if level == 0:
            fine.apply_bc()
            
            coarse.reset_solution()
        
        # Post-smoothing and residual update
        if nu2 > 0:
            fine.smooth(nu2)
        
        # Update residuals
        fine.update_residual()
        
        return fine.residual_norm() if level == 0 else None

    def set_rhs(self, p_rhs, vx_rhs, vy_rhs, grid):
        # Set the right-hand side for the fine grid
        grid.p_rhs[...] = p_rhs
        grid.vx_rhs[...] = vx_rhs
        grid.vy_rhs[...] = vy_rhs

    def set_guess(self, p_guess, vx_guess, vy_guess, grid):
        # Set the initial guess for the fine grid
        grid.p[...] = p_guess if p_guess is not None else 0.0
        grid.vx[...] = vx_guess if vx_guess is not None else 0.0
        grid.vy[...] = vy_guess if vy_guess is not None else 0.0

    def solve(self, p_rhs, vx_rhs, vy_rhs,
              max_cycles=50, tol=1e-7,
              nu1=3, nu2=3, gamma=1, pre_smooth=0,
              p_guess=None, vx_guess=None, vy_guess=None,
              p_ref=None):
        
        # Extract the fine grid
        fine = self.hierarchy[0]
        
        # Set the right-hand side for the fine grid
        self.set_rhs(p_rhs, vx_rhs, vy_rhs, fine)

        # Set the initial guess for the fine grid
        self.set_guess(p_guess, vx_guess, vy_guess, fine)
        
        # Run pre-smoothing cycles
        if pre_smooth > 0:
            fine.smooth(pre_smooth)
        
        # Fix anchor pressure if provided
        fine.p_ref = p_ref

        # Main multigrid loop
        for cycle in range(max_cycles):
            print(f"Cycle: {cycle}")
            norms = self.vcycle(0, nu1, nu2, gamma)
            if norms:
                p_res, vx_res, vy_res = norms
                print(f"Continuity residual: {p_res}")
                print(f"X-momentum residual: {vx_res}")
                print(f"Y-momentum residual: {vy_res}")
                if max(p_res, vx_res, vy_res) < tol:
                    break

        return fine.p, fine.vx, fine.vy
