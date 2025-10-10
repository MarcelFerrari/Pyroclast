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

from os import name
import numpy as np
from Pyroclast.profiling import timer
from Pyroclast.linalg import get_xp
from Pyroclast.arrayview import view

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

    def __init__(self, nx: int, ny:int, level: int, ctx: "Context", device: str) -> None:
        # Read context
        s, p, o = ctx

        # Device
        self.enable_gpu = p.get("enable_gpu", False)
        
        if self.enable_gpu:
            try:
                import cupy as cp
                self.cp = cp
            except ImportError:
                raise ImportError("GPU support requested but CuPy is not available.")

        self.device = device

        # Get array module
        self.xnp = get_xp(device)

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
        self.x = self.xnp.linspace(0, p.xsize + self.dx, self.nx1)
        self.y = self.xnp.linspace(0, p.ysize + self.dy, self.ny1)
        self.xvx = self.x
        self.yvx = self.y - self.dy / 2
        self.xvy = self.x - self.dx / 2
        self.yvy = self.y
        self.xp = self.x - self.dx / 2
        self.yp = self.y - self.dy / 2

        # Physical properties
        shape = (self.ny1, self.nx1)
        self.rho = self.xnp.zeros(shape)
        self.etab = self.xnp.zeros(shape)
        self.etap = self.xnp.zeros(shape)

        # Solution, RHS, residual arrays
        self.vx = self.xnp.zeros(shape)
        self.vy = self.xnp.zeros(shape)
        self.vx_new = self.xnp.zeros(shape)
        self.vy_new = self.xnp.zeros(shape)
        self.vx_rhs = self.xnp.zeros(shape)
        self.vy_rhs = self.xnp.zeros(shape)
        self.vx_res = self.xnp.zeros(shape)
        self.vy_res = self.xnp.zeros(shape)

        self._w = self.xnp.zeros(shape=shape, dtype=np.float64)
        self._w_gpu = None

        # Bind correct functions
        self.bind_device_functions()
    
    def bind_device_functions(self):
        if self.device == "gpu":
            from .implicit_operators.gpu import uzawa_vx_residual, uzawa_vy_residual
            self.uzawa_vx_residual = uzawa_vx_residual
            self.uzawa_vy_residual = uzawa_vy_residual

            from .smoothers.gpu import jacobi_velocity_smoother
            self.smoother = jacobi_velocity_smoother

            from .smoothers.gpu.bc import apply_vx_BC, apply_vy_BC
            self.apply_vx_BC = apply_vx_BC
            self.apply_vy_BC = apply_vy_BC

            from Pyroclast.solvers.multigrid.mg_routines_gpu import restrict_2D as restrict, \
                                                                    prolong_2D as prolong
            self.restrict = restrict
            self.prolong = prolong
        elif self.device == "cpu":
            from .implicit_operators.cpu import uzawa_vx_residual, uzawa_vy_residual
            self.uzawa_vx_residual = uzawa_vx_residual
            self.uzawa_vy_residual = uzawa_vy_residual

            from .smoothers.cpu import jacobi_velocity_smoother
            self.smoother = jacobi_velocity_smoother

            from .smoothers.cpu.bc import apply_vx_BC, apply_vy_BC
            self.apply_vx_BC = apply_vx_BC
            self.apply_vy_BC = apply_vy_BC

            try:
                from Pyroclast.turbo.mg_routines import restrict_2D as restrict, \
                                                         prolong_2D as prolong
            except ImportError:
                from Pyroclast.solvers.multigrid.mg_routines import restrict_2D as restrict, \
                                                                    prolong_2D as prolong
            
            self.restrict = restrict
            self.prolong = prolong
        else:
            raise ValueError(f"Unknown device: {self.device}")

    # Convenience wrappers for restriction/prolongation
    def get_weights(self, device: str):
        """
        Get weights for restriction/prolongation using only preallocated buffers.
        """
        # Weights are on same device (CPU or GPU) as self._w
        if self.device == device:
            return self._w
        
        # I am on CPU, but they want weights on GPU
        if self.device == "cpu" and device == "gpu":
            if self._w_gpu is None:
                self._w_gpu = self.cp.zeros_like(self._w)
            return self._w_gpu

        # I am on GPU, but they want weights on CPU
        # This is not allowed!
        # self.device == "gpu" and device == "cpu":
        raise RuntimeError("Cannot get array on CPU if I am on GPU.")
    
    @timer.time_function("Vcycle", "Update Residual")
    def update_residuals(self) -> None:
        self.vx_res = self.uzawa_vx_residual(
            self.nx1, self.ny1,
            self.dx, self.dy,
            self.etap, self.etab,
            self.vx, self.vy,
            self.vx_res, self.vx_rhs,
        )

        self.vy_res = self.uzawa_vy_residual(
            self.nx1, self.ny1,
            self.dx, self.dy,
            self.etap, self.etab,
            self.vx, self.vy,
            self.vy_res, self.vy_rhs,
        )

    @timer.time_function("Vcycle", "Smooth")
    def smooth(self, iterations: int) -> None:
        self.vx, self.vy = self.smoother(
            self.nx1, self.ny1,
            self.dx, self.dy,
            self.etap, self.etab,
            self.vx, self.vy,
            self.vx_new, self.vy_new,
            self.relax_v, self.BC,
            self.vx_rhs, self.vy_rhs,
            iterations
        )

    def apply_bc(self) -> None:
        self.apply_vx_BC(self.vx, self.BC)
        self.apply_vy_BC(self.vy, self.BC)

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
        self._weights.fill(0.0)

    @timer.time_function("Vcycle", "Restriction")
    def restrict_properties(self, fine: "Grid") -> None:
        # Get weights array on correct device
        _w = self.get_weights(fine.device)
        
        # Need to ensure that the arrays are on the correct device
        with view(self.xvy, intent="in", device=fine.device) as xvy_dev, \
             view(self.yvy, intent="in", device=fine.device) as yvy_dev, \
             view(self.rho, intent="out", device=fine.device) as rho_dev:
            fine.restrict(
                fine.nx1, fine.ny1,
                fine.xvy, fine.yvy, fine.rho,
                self.nx1, self.ny1,
                xvy_dev, yvy_dev,
                rho_dev, _w
            )

        with view(self.x, intent="in", device=fine.device) as x_dev, \
             view(self.y, intent="in", device=fine.device) as y_dev, \
             view(self.etab, intent="out", device=fine.device) as etab_dev:
            fine.restrict(
                fine.nx1, fine.ny1,
                fine.x, fine.y, fine.etab,
                self.nx1, self.ny1,
                x_dev, y_dev,
                etab_dev, _w
            )

        with view(self.xp, intent="in", device=fine.device) as xp_dev, \
             view(self.yp, intent="in", device=fine.device) as yp_dev, \
             view(self.etap, intent="out", device=fine.device) as etap_dev:
            fine.restrict(
                fine.nx1, fine.ny1,
                fine.xp, fine.yp, fine.etap,
                self.nx1, self.ny1,
                xp_dev, yp_dev,
                etap_dev, _w
            )

    @timer.time_function("Vcycle", "Restriction")
    def restrict_residuals(self, fine: "Grid") -> None:
        # Get weights array on correct device
        _w = self.get_weights(fine.device)

        # Need to ensure that the arrays are on the correct device
        with view(self.xvx, intent="in", device=fine.device) as xvx_dev, \
             view(self.yvx, intent="in", device=fine.device) as yvx_dev, \
             view(self.vx_rhs, intent="out", device=fine.device) as vx_rhs_dev:
            fine.restrict(
                fine.nx1, fine.ny1,
                fine.xvx, fine.yvx, fine.vx_res,
                self.nx1, self.ny1,
                xvx_dev, yvx_dev,
                vx_rhs_dev, _w
            )

        with view(self.xvy, intent="in", device=fine.device) as xvy_dev, \
             view(self.yvy, intent="in", device=fine.device) as yvy_dev, \
             view(self.vy_rhs, intent="out", device=fine.device) as vy_rhs_dev:
            fine.restrict(
                fine.nx1, fine.ny1,
                fine.xvy, fine.yvy, fine.vy_res,
                self.nx1, self.ny1,
                xvy_dev, yvy_dev,
                vy_rhs_dev, _w
            )

    @timer.time_function("Vcycle", "Prolongation")
    def prolong_correction(self, coarse: "Grid") -> None:
        # Need to move coarse grid arrays to correct device if needed
        with view(coarse.xvx, intent="in", device=self.device) as xvx_dev, \
             view(coarse.yvx, intent="in", device=self.device) as yvx_dev, \
             view(coarse.vx, intent="in", device=self.device) as vx_dev:
            
            self.prolong(
                coarse.nx1, coarse.ny1,
                xvx_dev, yvx_dev,
                vx_dev,
                self.nx1, self.ny1,
                self.xvx, self.yvx,
                self.vx_res # Store correction in residual array
            )
        
        # Apply correction
        self.vx += self.vx_res
        
        with view(coarse.xvy, intent="in", device=self.device) as xvy_dev, \
             view(coarse.yvy, intent="in", device=self.device) as yvy_dev, \
             view(coarse.vy, intent="in", device=self.device) as vy_dev:
            self.prolong(
                coarse.nx1, coarse.ny1,
                xvy_dev, yvy_dev,
                vy_dev,
                self.nx1, self.ny1,
                self.xvy, self.yvy,
                self.vy_res # Store correction in residual array
            )

        # Apply correction
        self.vy += self.vy_res