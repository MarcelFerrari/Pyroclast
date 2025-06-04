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
from .smoother import velocity_smoother
from .mg_routines import restrict, prolong, uzawa_vx_residual, uzawa_vy_residual
from .utils import apply_vx_BC, apply_vy_BC


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
        self.vx = np.zeros(shape)
        self.vy = np.zeros(shape)
        self.vx_rhs = np.zeros(shape)
        self.vy_rhs = np.zeros(shape)
        self.vx_res = np.zeros(shape)
        self.vy_res = np.zeros(shape)

        # Boundary conditions and relaxation
        self.BC = params.BC
        self.relax_v = params.get("relax_v", 0.7)
            
    @timer.time_function("Vcycle", "Update Residual")
    def update_residual(self):
        self.vx_res = uzawa_vx_residual(
                                        self.nx1, self.ny1,
                                        self.dx, self.dy,
                                        self.etap, self.etab,
                                        self.vx, self.vy,
                                        self.vx_res, self.vx_rhs,
                                    )

        self.vy_res = uzawa_vy_residual(
                                        self.nx1, self.ny1,
                                        self.dx, self.dy,
                                        self.etap, self.etab,
                                        self.vx, self.vy,
                                        self.vy_res, self.vy_rhs,
                                    )

    def residual_norms(self):
        N = np.sqrt(self.nx1 * self.ny1)
        vx_res_rmse = np.linalg.norm(self.vx_res) / N
        vy_res_rmse = np.linalg.norm(self.vy_res) / N
        return vx_res_rmse, vy_res_rmse
    
    @timer.time_function("Vcycle", "Smooth")
    def smooth(self, iterations):
        # Smooth the velocity field
        self.vx, self.vy = velocity_smoother(self.nx1, self.ny1,
                                            self.dx, self.dy,
                                            self.etap, self.etab,
                                            self.vx, self.vy,
                                            self.relax_v, self.BC,
                                            self.vx_rhs, self.vy_rhs, iterations)

    def apply_bc(self):
        apply_vx_BC(self.vx, self.BC)
        apply_vy_BC(self.vy, self.BC)

    def reset_solution(self):
        """Reset the solution and residual arrays to zero y but not the material properties."""
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
        self.vx_rhs = restrict(
            fine.xvx, fine.yvx, fine.vx_res,
            self.xvx, self.yvx,
        )
        self.vy_rhs = restrict(
            fine.xvy, fine.yvy, fine.vy_res,
            self.xvy, self.yvy,
        )

    @timer.time_function("Vcycle", "Prolongation")
    def prolong_correction(self, coarse):
        self.vx += prolong(
            self.xvx, self.yvx,
            coarse.xvx, coarse.yvx,
            coarse.vx,
        )
        self.vy += prolong(
            self.xvy, self.yvy,
            coarse.xvy, coarse.yvy,
            coarse.vy,
        )

    def viscosity_contrast(self):
        """Return the ratio of maximum to minimum viscosity on this grid."""
        etabmin = np.nanmin(self.etab[:-1, :-1])
        etabmax = np.nanmax(self.etab[:-1, :-1])
        etapmin = np.nanmin(self.etap[:-1, :-1])
        etapmax = np.nanmax(self.etap[:-1, :-1])
        return max(etabmax, etapmax) / (1.0 + min(etabmin, etapmin))

