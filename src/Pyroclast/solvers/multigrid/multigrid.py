"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: base.py
Description: Defines a reusable base class implementing the generic
             multigrid V-cycle algorithm.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from abc import ABC, abstractmethod
from typing import Sequence, Any


class BaseMultigrid(ABC):
    """Generic multigrid solver implementing a recursive V-cycle.

    Parameters
    ----------
    hierarchy:
        Ordered sequence of grid levels from finest (index 0) to coarsest.
        Each grid level must implement the following methods:

        - ``smooth(iterations)`` – apply the smoother for ``iterations`` steps.
        - ``update_residual()`` – compute the residual of the current solution.
        - ``restrict_residuals(fine_grid)`` – restrict residuals from ``fine_grid``.
        - ``prolong_correction(coarse_grid)`` – prolong correction from ``coarse_grid``.
        - ``apply_bc()`` – enforce boundary conditions on the solution variables.
        - ``reset_solution()`` – reset solution values to zero (used after prolongation).

    scaling:
        Factor by which the number of smoothing iterations is scaled when
        moving to coarser grids. Defaults to ``2.0``.
    """

    def __init__(self, hierarchy: Sequence[Any], scaling: float = 2.0) -> None:
        self.hierarchy = hierarchy
        self.scaling = scaling

    def vcycle(self, level: int, nu1: int, nu2: int) -> None:
        """Perform a single V-cycle starting at ``level``.

        Parameters
        ----------
        level:
            Index of the current grid level.
        nu1:
            Number of pre-smoothing iterations.
        nu2:
            Number of post-smoothing iterations.
        """
        fine = self.hierarchy[level]

        if nu1 > 0:
            self.pre_smooth(fine, nu1)

        if level + 1 < len(self.hierarchy):
            coarse = self.hierarchy[level + 1]

            self.compute_residual(fine)
            self.restrict(fine, coarse)

            self.vcycle(level + 1, int(nu1 * self.scaling), int(nu2 * self.scaling))

            self.prolong(coarse, fine)
            self.apply_bc(fine)
            self.reset(coarse)

        if nu2 > 0:
            self.post_smooth(fine, nu2)

    @abstractmethod
    def pre_smooth(self, grid: Any, iterations: int) -> None:
        """Apply pre-smoothing to ``grid`` for ``iterations`` steps."""

    @abstractmethod
    def post_smooth(self, grid: Any, iterations: int) -> None:
        """Apply post-smoothing to ``grid`` for ``iterations`` steps."""

    @abstractmethod
    def compute_residual(self, grid: Any) -> None:
        """Update the residual stored in ``grid``."""

    @abstractmethod
    def restrict(self, fine: Any, coarse: Any) -> None:
        """Restrict residuals from ``fine`` grid to ``coarse`` grid."""

    @abstractmethod
    def prolong(self, coarse: Any, fine: Any) -> None:
        """Prolong correction from ``coarse`` grid to ``fine`` grid."""

    @abstractmethod
    def apply_bc(self, grid: Any) -> None:
        """Apply boundary conditions on ``grid``."""

    @abstractmethod
    def reset(self, grid: Any) -> None:
        """Reset the solution values on ``grid``."""

    @abstractmethod
    def extract_solution(self, grid: Any):
        """Return the solution stored on ``grid``."""