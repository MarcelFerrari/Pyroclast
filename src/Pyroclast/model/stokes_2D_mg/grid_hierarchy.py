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
from Pyroclast.logging import get_logger

logger = get_logger(__name__)

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

        logger.info(f"Grid Hierarchy: {self.nlevels} levels, scaling {scaling:.2f}")
        logger.info(f"Fine grid: {base.ny1} x {base.nx1}")
    
        # Build coarse grids and propagate properties
        for lvl in range(1, self.nlevels):
            prev = self.levels[-1]
            nx_coarse = int(prev.nx / scaling)
            ny_coarse = int(prev.ny / scaling)
            coarse = Grid(ny_coarse, nx_coarse, lvl, ctx)
            coarse.restrict_properties(prev)
            self.levels.append(coarse)
            logger.info(f"Coarse grid {lvl}: {coarse.ny1} x {coarse.nx1}")

    def __getitem__(self, idx):
        return self.levels[idx]

    def __len__(self):
        return len(self.levels)

