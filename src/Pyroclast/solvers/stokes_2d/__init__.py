"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: solvers/stokes_2d/__init__.py
Description: Package exports for Stokes solvers and refinement utilities.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

"""Stokes-specific multigrid solver wrappers."""

from .uzawa_solver import UzawaSolver
from .iterative_refinement import IterativeRefinement

__all__ = ["UzawaSolver", "iterative_refinement"]
