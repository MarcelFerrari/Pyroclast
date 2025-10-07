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
import math
from typing import Optional

from Pyroclast.context import ContextNamespace, Context
from Pyroclast.logging import get_logger
from .grid import Grid

logger = get_logger(__name__)

class GridHierarchy:
    """
    Builds and stores grid levels from fine to coarse.
    """
    def __init__(self, ctx: Context, nlevels, scaling):
        state: ContextNamespace
        params: ContextNamespace
        _opts: ContextNamespace

        state, params, _opts = ctx

        # Init grid
        base = Grid(state.nx1 - 1, state.ny1 - 1, 0, ctx)
        
        # Copy over material properties
        base.rho[:,:] = state.rho
        base.etab[:,:] = state.etab
        base.etap[:,:] = state.etap

        # Initialize coarse levels
        self.nlevels = nlevels
        self.levels = [base]
        logger.info(f"Grid Hierarchy: {self.nlevels} levels, scaling {scaling:.2f}")
        logger.info(f"Fine grid: {base.ny1} x {base.nx1}")
    
        # Build coarse grids and propagate properties
        for lvl in range(1, self.nlevels):
            prev = self.levels[-1]
            # Half the number of cells
            nx_coarse = int((prev.nx - 1) / scaling) + 1
            ny_coarse = int((prev.ny - 1) / scaling) + 1 
            coarse = Grid(ny_coarse, nx_coarse, lvl, ctx)
            coarse.restrict_properties(prev)
            self.levels.append(coarse)
            logger.info(f"Coarse grid {lvl}: {coarse.ny1} x {coarse.nx1}")

    def __getitem__(self, idx):
        return self.levels[idx]

    def __len__(self):
        return len(self.levels)

    def reset(self):
        # Reset all grid levels
        for level in self.levels:
            level.reset()

