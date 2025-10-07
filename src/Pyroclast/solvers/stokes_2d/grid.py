"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: grid.py
Description: This file implements the grid class for the multigrid method
              for the Stokes flow and continuity equations in 2D.
             

Author: Marcel Ferrari, Alexander Sotoudeh
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
import importlib

import numpy as np

from Pyroclast.profiling import timer

try:
    from Pyroclast.turbo.mg_routines import restrict_2D as restrict, prolong_2D as prolong
    print("Using Turbo mg_routines.")
except ImportError:
    from Pyroclast.solvers.multigrid.mg_routines import restrict_2D as restrict, prolong_2D as prolong
    print("Turbo mg_routines not found, using pure Python version.")


from .smoother import velocity_jacobi_smoother
from .bc import cpu_apply_vy_BC, cpu_apply_vx_BC, gpu_apply_vx_bc_kernel, gpu_apply_vy_bc_kernel
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

    is_gpu: bool

    # -------------------------------------------------------------------------

    def __init__(self, nx: int, ny:int, level: int, ctx: "Context", is_gpu: bool) -> None:
        self.is_gpu = is_gpu

        xp = importlib.import_module("cupy") if self.is_gpu else importlib.import_module("numpy")

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
        self.x = xp.linspace(0, p.xsize + self.dx, self.nx1)
        self.y = xp.linspace(0, p.ysize + self.dy, self.ny1)
        self.xvx = self.x
        self.yvx = self.y - self.dy / 2
        self.xvy = self.x - self.dx / 2
        self.yvy = self.y
        self.xp = self.x - self.dx / 2
        self.yp = self.y - self.dy / 2

        # Physical properties
        shape = (self.ny1, self.nx1)
        self.rho = xp.zeros(shape)
        self.etab = xp.zeros(shape)
        self.etap = xp.zeros(shape)

        # Solution, RHS, residual arrays
        self.vx = xp.zeros(shape)
        self.vy = xp.zeros(shape)
        self.vx_new = xp.zeros(shape)
        self.vy_new = xp.zeros(shape)
        self.vx_rhs = xp.zeros(shape)
        self.vy_rhs = xp.zeros(shape)
        self.vx_res = xp.zeros(shape)
        self.vy_res = xp.zeros(shape)

        # Weights for interpolation (to avoid reallocating)
        self._w = xp.zeros((self.ny1, self.nx1))
        self.__gpu_acc = None
        self.__gpu_weights = None

    def allocate_temp_arrays(self):
        """
        Allocate temporary arrays for transition between gpu and cpu
        """
        import cupy as cp

        self.__gpu_acc = cp.ndarray(shape=(self.ny1, self.nx1), dtype=np.float64)
        self.__gpu_weights = cp.ndarray(shape=(self.ny1, self.nx1), dtype=np.float64)

    def _copy_to_device(self, attr: str):
        """
        Internal function for prolong and restrict. Copies to a reused and constantly allocated array on gpu.
        """
        import cupy as cp

        if attr not in ("rho", "etab", "etap", "vx_rhs", "vy_rhs"):
            raise ValueError("Unsupported source array")

        if self.__gpu_acc is None:
            raise ValueError("Implementation incorrect. temp arrays needed for transitional grid")

        source = getattr(self, attr)

        self.__gpu_acc: cp.ndarray
        cp.cuda.runtime.memcpy(self.__gpu_acc.data.ptr,
                               source.ctypes.data,
                               source.nbytes,
                               cp.cuda.runtime.memcpyHostToDevice)
        # self.__gpu_acc.set(source)

        return self.__gpu_acc

    def outer_to_device(self, source: np.ndarray, attr: str):
        """
        Copy initial values to the gpu arrays
        """
        import cupy as cp

        if attr not in ("rho", "etab", "etap", "vx_rhs", "vy_rhs", "vx", "vy", "vx_new", "vy_new", "vx_res", "vx_res"):
            raise ValueError("Unsupported source array")

        target: cp.ndarray = getattr(self, attr)
        cp.cuda.runtime.memcpy(target.data.ptr,
                               source.ctypes.data,
                               source.nbytes,
                               cp.cuda.runtime.memcpyHostToDevice)
        # target.set(source)

    def _copy_from_device(self, attr: str):
        """
        Internal function for prolong and restrict. Copies from a reused and constantly allocated array on gpu.
        """
        import cupy as cp

        if attr not in ("rho", "etab", "etap", "vx_rhs", "vy_rhs"):
            raise ValueError("Unsupported source array")

        if self.__gpu_acc is None:
            raise ValueError("Implementation incorrect. temp arrays needed for transitional grid")

        self.__gpu_acc: cp.ndarray
        tgt: np.ndarray = getattr(self, attr)

        # Ok, we're going to use some scary functions.
        cp.cuda.runtime.memcpy(tgt.ctypes.data,
                               self.__gpu_acc.data.ptr,
                               self.__gpu_acc.nbytes,
                               cp.cuda.runtime.memcpyDeviceToHost)
        # self.__gpu_acc.get(tgt)

    @timer.time_function("Vcycle", "Update Residual")
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

    @timer.time_function("Vcycle", "Smooth")
    def smooth(self, iterations: int) -> None:
        self.vx, self.vy = velocity_jacobi_smoother(
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
        cpu_apply_vx_BC(self.vx, self.BC)
        cpu_apply_vy_BC(self.vy, self.BC)

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
        self._w.fill(0.0)

    @timer.time_function("Vcycle", "Restriction")
    def restrict_properties(self, fine: "Grid") -> None:
        self.rho = restrict(
            fine.nx1, fine.ny1,
            fine.xvy, fine.yvy, fine.rho,
            self.nx1, self.ny1,
            self.xvy, self.yvy,
            self.rho, self._w
        )
        self.etab = restrict(
            fine.nx1, fine.ny1,
            fine.x, fine.y, fine.etab,
            self.nx1, self.ny1,
            self.x, self.y,
            self.etab, self._w
        )
        self.etap = restrict(
            fine.nx1, fine.ny1,
            fine.xp, fine.yp, fine.etap,
            self.nx1, self.ny1,
            self.xp, self.yp,
            self.etap, self._w
        )

    @timer.time_function("Vcycle", "Restriction")
    def restrict_residuals(self, fine: "Grid") -> None:
        self.vx_rhs = restrict(
            fine.nx1, fine.ny1,
            fine.xvx, fine.yvx, fine.vx_res,
            self.nx1, self.ny1,
            self.xvx, self.yvx,
            self.vx_rhs, self._w
        )
        self.vy_rhs = restrict(
            fine.nx1, fine.ny1,
            fine.xvy, fine.yvy, fine.vy_res,
            self.nx1, self.ny1,
            self.xvy, self.yvy,
            self.vy_rhs, self._w
        )

    @timer.time_function("Vcycle", "Prolongation")
    def prolong_correction(self, coarse: "Grid") -> None:
        self.vx += prolong(
            coarse.nx1, coarse.ny1,
            coarse.xvx, coarse.yvx,
            coarse.vx,
            self.nx1, self.ny1,
            self.xvx, self.yvx,
            self.vx_res # Store correction in residual array
        )
        self.vy += prolong(
            coarse.nx1, coarse.ny1,
            coarse.xvy, coarse.yvy,
            coarse.vy,
            self.nx1, self.ny1,
            self.xvy, self.yvy,
            self.vy_res # Store correction in residual array
        )
