"""
Stokes-specific multigrid implementation.

This module provides :class:`StokesMultigrid`, a concrete subclass of
:class:`~Pyroclast.solvers.multigrid.BaseMultigrid` that wires the generic
V-cycle logic to the Stokes grid hierarchy and smoothing operators.
"""

from __future__ import annotations

from Pyroclast.solvers.multigrid import BaseMultigrid
from Pyroclast.model.stokes_2D_mg.grid_hierarchy import GridHierarchy


class StokesMultigrid(BaseMultigrid):
    """Multigrid solver tailored for the 2-D Stokes system."""

    def __init__(self, ctx, levels: int, scaling: float = 2.0) -> None:
        hierarchy = GridHierarchy(ctx, levels, scaling)
        super().__init__(hierarchy, scaling)

    def pre_smooth(self, grid, iterations: int) -> None:
        grid.smooth(iterations)

    def post_smooth(self, grid, iterations: int) -> None:
        grid.smooth(iterations)

    def compute_residual(self, grid) -> None:
        grid.update_residual()

    def restrict(self, fine, coarse) -> None:
        coarse.restrict_residuals(fine)

    def prolong(self, coarse, fine) -> None:
        fine.prolong_correction(coarse)

    def apply_bc(self, grid) -> None:
        grid.apply_bc()

    def reset(self, grid) -> None:
        grid.reset_solution()

    def extract_solution(self, grid):
        return grid.vx, grid.vy
