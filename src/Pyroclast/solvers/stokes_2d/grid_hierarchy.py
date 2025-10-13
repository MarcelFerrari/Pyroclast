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
from Pyroclast.logging import get_logger

from .grid import Grid

logger = get_logger(__name__)

class GridHierarchy:
    """
    Builds and stores grid levels from fine to coarse.
    """
    def __init__(self, ctx, nlevels, scaling):
        state, params, _opts = ctx

        self.gpu_enable = params.get("enable_gpu", False)
        self.gpu_threshold: int = params.get("gpu_threshold", 1024)

        # Init grid
        base = Grid(state.nx1 - 1, state.ny1 - 1, 0, ctx, \
                    device = self.get_device(state.nx1 - 1, state.ny1 - 1))
        
        # Initialize coarse levels
        self.nlevels = nlevels
        self.levels = [base]
        logger.info(f"Grid Hierarchy: {self.nlevels} levels, scaling {scaling:.2f}")
        logger.info(f"Fine grid: {base.ny1} x {base.nx1} on device {base.device}")
    
        # Build coarse grids and propagate properties
        for lvl in range(1, self.nlevels):
            prev = self.levels[-1]
            
            # Half the number of cells
            nx_coarse = int((prev.nx - 1) / scaling) + 1
            ny_coarse = int((prev.ny - 1) / scaling) + 1
            coarse_device = self.get_device(nx_coarse, ny_coarse)
            coarse = Grid(ny_coarse, nx_coarse, lvl, ctx, device=coarse_device)    
            self.levels.append(coarse)
            logger.info(f"Coarse grid {lvl}: {coarse.ny1} x {coarse.nx1} on device {coarse.device}")

    def set_properties(self, etab, etap, rho):
        """Set new material properties on the finest grid."""
        fine = self.levels[0]
        fine.etab[...] = etab
        fine.etap[...] = etap
        fine.rho[...] = rho

        # Propagate to coarse levels
        for lvl in range(1, self.nlevels):
            prev = self.levels[lvl - 1]
            coarse = self.levels[lvl]
            coarse.restrict_properties(prev)

    def get_device(self, nx, ny):
        if not self.gpu_enable or \
           np.sqrt(nx * ny) < self.gpu_threshold:
            return "cpu"
        else:
            return "gpu"
    
    def __getitem__(self, idx):
        return self.levels[idx]

    def __len__(self):
        return len(self.levels)

    def reset(self):
        # Reset all grid levels
        for level in self.levels:
            level.reset()

