"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: multigrid.py
Description: This file implements the multigrid method used for the velocity
             solve of the inexact Uzawa iteration.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
import numpy as np
from Pyroclast.profiling import timer
from .grid_hierarchy import GridHierarchy


class Multigrid:
    def __init__(self, ctx, levels, scaling=2.0):
        self.scaling = scaling
        self.hierarchy = GridHierarchy(ctx, levels, scaling)

    def vcycle(self, level, nu1, nu2):
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
            self.vcycle(level + 1, nu1*self.scaling, nu2*self.scaling)
            
            # Prolongate correction to fine grid
            fine.prolong_correction(coarse)
            
            fine.apply_bc()
            
            coarse.reset_solution()
        
        # Post-smoothing and residual update
        if nu2 > 0:
            fine.smooth(nu2)
        
        # Update residuals
        fine.update_residual()

        return fine.vx, fine.vy
