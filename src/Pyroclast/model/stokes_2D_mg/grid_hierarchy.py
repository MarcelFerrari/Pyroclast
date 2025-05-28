"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: grid_hierarchy.py
Description: This file implements the grid hierarchy for the multigrid method.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np
from .grid import Grid

class GridHierarchy:
    """
    Builds and stores grid levels from fine to coarse.
    """
    def __init__(self, ctx, nlevels, scaling):
        state, params, _opts = ctx

        # Fine grid
        base = Grid(params.ny, params.nx, 0, ctx)
        base.rho[:] = state.rho
        base.etab[:] = state.etab
        base.etap[:] = state.etap
        self.nlevels = nlevels
        self.levels = [base]

        print(f"Grid Hierarchy: {self.nlevels} levels, scaling {scaling:.2f}")
        print(f"Fine grid: {base.ny1} x {base.nx1}")
    
        # Build coarse grids
        for lvl in range(1, self.nlevels):
            prev = self.levels[-1]
            nx_coarse = int(prev.nx / scaling)
            ny_coarse = int(prev.ny / scaling)
            coarse = Grid(ny_coarse, nx_coarse, lvl, ctx)
            self.levels.append(coarse)
            print(f"Coarse grid {lvl}: {coarse.ny1} x {coarse.nx1}")

        # Optional viscosity rescaling
        self.scale_viscosity = params.get("eta_scaling", False)# and state.iteration == 0
        if self.scale_viscosity:
            # Store original viscosity for optional rescaling
            self.etab_original = state.etab # Reference to original viscosity
            self.etap_original = state.etap 
            self.eta_cycle = 0 # cycle counter
            self.eta_counter = 0 # number of rescales done
            self.eta_rescale_interval = params.eta_cycle_interval
            self.eta_ncycles = params.eta_ncycles
            self.eta_theta = 0.0
            self.eta_theta_step = 1.0 / (self.eta_ncycles-1)            
            self.etab_min = np.nanmin(self.etab_original[:-1, :-1])
            self.etap_min = np.nanmin(self.etap_original[:-1, :-1])

            # Adjust viscosity if scaling is enabled
            self.recompute_viscosity()

        self.propagate_properties()

    def done_rescaling(self):
        """
        Check if viscosity rescaling is done.
        """
        if not self.scale_viscosity:
            return True
        
        return self.eta_counter >= self.eta_ncycles
    
    def propagate_properties(self):
        # Propagate to coarse grids
        print("Propagating viscosity to coarse grids")
        for lvl in range(1, self.nlevels):
            prev = self.levels[lvl-1]
            coarse = self.levels[lvl]
            coarse.restrict_properties(prev)

    def recompute_viscosity(self):
        print(f"Rescaling viscosity: {self.eta_theta:.2f}")
        # Log-space interpolation
        self.eta_theta = min(self.eta_theta, 1.0)
        
        fine = self.levels[0]

        # Linear interpolation of viscosity
        fine.etab = self.etab_min * (1.0 - self.eta_theta) + \
                    self.etab_original * self.eta_theta
        fine.etap = self.etap_min * (1.0 - self.eta_theta) + \
                    self.etap_original * self.eta_theta
        
        # Remove Nans
        fine.etab = np.nan_to_num(fine.etab, copy=False)
        fine.etap = np.nan_to_num(fine.etap, copy=False)

    def update_viscosity(self):
        """
        Rescale viscosity to gradually increase the contrast by increasing eta_max_current,
        keeping eta_comp_min fixed. Only triggers if viscosity scaling is enabled and enough
        cycles have passed.
        """
        if not self.scale_viscosity:
            return False

        self.eta_cycle += 1

        if self.eta_cycle < self.eta_rescale_interval:
            return False
        
        if self.eta_counter >= self.eta_ncycles:
            return False
        
        self.eta_theta += self.eta_theta_step

        # Last cycle: clamp to max
        if self.eta_counter == self.eta_ncycles - 1:
            self.eta_theta = 1.0

        # Recompute viscosity
        self.recompute_viscosity()

        # Propagate material properties to coarse grids
        self.propagate_properties()

        self.eta_cycle = 0  # reset interval counter
        self.eta_counter += 1
        return True

    def __getitem__(self, idx):
        return self.levels[idx]

    def __len__(self):
        return len(self.levels)