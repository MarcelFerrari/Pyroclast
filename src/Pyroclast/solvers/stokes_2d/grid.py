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
from Pyroclast.context import ContextNamespace
from Pyroclast.solvers.multigrid import restrict_2D as restrict, prolong_2D as prolong

from .smoother import velocity_smoother
from .bc import apply_vx_BC, apply_vy_BC
from .implicit_operators import uzawa_vx_residual, uzawa_vy_residual

class Grid:
    """
    Single multigrid level: geometry, solution, RHS, residuals,
    and operations for smoothing, residual update, restriction, and prolongation.

    Required attributes (must be set by child classes before calling methods here):

    Geometry / coordinates (staggered + cell-centered):
        x,  y         : Coordinates of nodal points of size
        xp, yp        : Coordinates of pressure points
        xvx, yvx      : Coordinates of vx points
        xvy, yvy      : Coordinates of vy points

    Material / fields:
        etab          : (ny1, nx1)    viscosity in basic nodes
        etap          : (ny1, nx1)    viscosity in pressure nodes
        rho           : (ny1, nx1)    density in vy nodes

    Unknowns and operators:
        vx, vy        : velocity fields on their respective staggered grids
        vx_rhs, vy_rhs: RHS arrays aligned with vx, vy
        vx_res, vy_res: residual arrays aligned with vx, vy
    """

    # --- Attributes expected to be assigned by subclasses ---------------------
    # Geometry
    nx1: int; ny1: int
    dx: float; dy: float

    x: np.ndarray; y: np.ndarray
    xp: np.ndarray; yp: np.ndarray
    xvx: np.ndarray; yvx: np.ndarray
    xvy: np.ndarray; yvy: np.ndarray

    # Material fields
    etab: np.ndarray; etap: np.ndarray; rho: np.ndarray

    # Unknowns, RHS, residuals
    vx: np.ndarray; vy: np.ndarray
    vx_rhs: np.ndarray; vy_rhs: np.ndarray
    vx_res: np.ndarray; vy_res: np.ndarray
    # -------------------------------------------------------------------------

    def __init__(self, nx: int, ny:int, level: int, ctx: "Context") -> None:
        # Read context
        s, p, o = ctx

        # Parameters
        self.BC = p.BC  # type depends on your BC convention
        self.relax_v = p.get("relax_v", 0.7)

        # Multigrid level
        self.level = level
        self.nx = nx
        self.ny = ny
        self.nx1 = nx + 1
        self.ny1 = ny + 1

        # Grid spacing
        self.dx = p.xsize / (nx - 1)
        self.dy = p.ysize / (ny - 1)

        # Coordinates for staggered grid
        self.x = np.linspace(0, p.xsize + self.dx, self.nx1)
        self.y = np.linspace(0, p.ysize + self.dy, self.ny1)
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

    # @timer.time_function("Vcycle", "Update Residual")
    def update_residuals(self) -> None:
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

    # @timer.time_function("Vcycle", "Smooth")
    def smooth(self, iterations: int) -> None:
        self.vx, self.vy = velocity_smoother(
            self.nx1, self.ny1,
            self.dx, self.dy,
            self.etap, self.etab,
            self.vx, self.vy,
            self.relax_v, self.BC,
            self.vx_rhs, self.vy_rhs, iterations
        )

    def apply_bc(self) -> None:
        apply_vx_BC(self.vx, self.BC)
        apply_vy_BC(self.vy, self.BC)

    def reset_solution(self) -> None:
        """Reset the solution fields to zero (but not material properties)."""
        self.vx.fill(0.0)
        self.vy.fill(0.0)
    
    def reset(self) -> None:
        """Reset all arrays to zero"""
        self.vx.fill(0.0)
        self.vy.fill(0.0)
        self.vx_rhs.fill(0.0)
        self.vy_rhs.fill(0.0)
        self.vx_res.fill(0.0)
        self.vy_res.fill(0.0)

    @timer.time_function("Vcycle", "Restriction")
    def restrict_properties(self, fine: "BaseGrid") -> None:
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
    def restrict_residuals(self, fine: "BaseGrid") -> None:
        self.vx_rhs = restrict(
            fine.xvx, fine.yvx, fine.vx_res,
            self.xvx, self.yvx,
        )
        self.vy_rhs = restrict(
            fine.xvy, fine.yvy, fine.vy_res,
            self.xvy, self.yvy,
        )

    @timer.time_function("Vcycle", "Prolongation")
    def prolong_correction(self, coarse: "BaseGrid") -> None:
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