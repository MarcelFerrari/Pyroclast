"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: grid.py
Description: This file implements the grid class for the multigrid method
              for the Stokes flow and continuity equations in 2D.
             

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np
from Pyroclast.profiling import timer
from .utils import apply_BC
from .smoother import uzawa
from .implicit_operators import vx_residual, vy_residual, p_residual
from .mg_routines import restrict, prolong


class Grid:
    """
    Single multigrid level: geometry, solution, RHS, residuals,
    and operations for smoothing, residual update,
    restriction, and prolongation.
    """

    def __init__(self, ny, nx, level, ctx):
        state, params, _opts = ctx
        self.level = level
        self.nx = nx
        self.ny = ny
        self.nx1 = nx + 1
        self.ny1 = ny + 1

        # Grid spacing
        self.dx = params.xsize / (nx - 1)
        self.dy = params.ysize / (ny - 1)

        # Coordinates for staggered grid
        self.x = np.linspace(0, params.xsize + self.dx, self.nx1)
        self.y = np.linspace(0, params.ysize + self.dy, self.ny1)
        self.xvx = self.x
        self.yvx = self.y - self.dy / 2
        self.xvy = self.x - self.dx / 2
        self.yvy = self.y
        self.xp = self.x - self.dx / 2
        self.yp = self.y - self.dy / 2

        # Physical properties
        shape = (self.ny1, self.nx1)
        self.rho = np.zeros(shape)
        self.etab = np.zeros(shape)
        self.etap = np.zeros(shape)

        # Solution, RHS, residual arrays
        self.p = np.zeros(shape)
        self.vx = np.zeros(shape)
        self.vy = np.zeros(shape)
        self.p_rhs = np.zeros(shape)
        self.vx_rhs = np.zeros(shape)
        self.vy_rhs = np.zeros(shape)
        self.p_res = np.zeros(shape)
        self.vx_res = np.zeros(shape)
        self.vy_res = np.zeros(shape)

        # Boundary conditions and relaxation
        self.BC = params.BC
        self.relax_v = 0.9
        self.relax_p = 0.9
        self.p_ref = None

    @timer.time_function("Vcycle", "Update Residual")
    def update_residual(self):
        vx_residual(
            self.nx1, self.ny1,
            self.dx, self.dy,
            self.etap, self.etab,
            self.vx, self.vy, self.p,
            self.vx_res, self.vx_rhs,
        )
        vy_residual(
            self.nx1, self.ny1,
            self.dx, self.dy,
            self.etap, self.etab,
            self.vx, self.vy, self.p,
            self.vy_res, self.vy_rhs,
        )
        p_residual(
            self.nx1, self.ny1,
            self.dx, self.dy,
            self.vx, self.vy,
            self.p_res, self.p_rhs,
        )

    def residual_norm(self):
        scale = (self.nx1 * self.ny1) ** 0.5
        p_norm = np.linalg.norm(self.p_res) / scale
        vx_norm = np.linalg.norm(self.vx_res) / scale
        vy_norm = np.linalg.norm(self.vy_res) / scale
        return p_norm, vx_norm, vy_norm

    @timer.time_function("Vcycle", "Smooth")
    def smooth(self, iterations):
        uzawa(
            self.nx1, self.ny1,
            self.dx, self.dy,
            self.etap, self.etab,
            self.vx, self.vy, self.p,
            self.relax_v, self.relax_p,
            self.p_ref, self.BC,
            self.p_rhs, self.vx_rhs, self.vy_rhs,
            iterations,
        )

    def apply_bc(self):
        apply_BC(self.p, self.vx, self.vy, self.BC)

    def reset_solution(self):
        """Reset the solution and residual arrays to zero y but not the material properties."""
        self.p.fill(0.0)
        self.vx.fill(0.0)
        self.vy.fill(0.0)

    def restrict_properties(self, fine):
        self.rho = restrict(
            fine.yvx, fine.yvy, fine.rho,
            self.xvx, self.yvx,
        )
        self.etab = restrict(
            fine.x, fine.y, fine.etab,
            self.x, self.y,
        )
        self.etap = restrict(
            fine.xp, fine.yp, fine.etap,
            self.xp, self.yp,
        )

    @timer.time_function("Vcycle", "Restriction")
    def restrict_residuals(self, fine):
        self.p_rhs = restrict(
            fine.xp, fine.yp,
            fine.p_res * fine.etap,
            self.xp, self.yp,
        ) / self.etap
        self.vx_rhs = restrict(
            fine.xvx, fine.yvx, fine.vx_res,
            self.xvx, self.yvx,
        )
        self.vy_rhs = restrict(
            fine.xvy, fine.yvy, fine.vy_res,
            self.xvy, self.yvy,
        )

    @timer.time_function("Vcycle", "Prolongation")
    def prolong_correction(self, fine):
        fine.p += prolong(
            fine.xp, fine.yp,
            self.xp, self.yp,
            self.p * self.etap,
        ) / fine.etap
        fine.vx += prolong(
            fine.xvx, fine.yvx,
            self.xvx, self.yvx,
            self.vx,
        )
        fine.vy += prolong(
            fine.xvy, fine.yvy,
            self.xvy, self.yvy,
            self.vy,
        )