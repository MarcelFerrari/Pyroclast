"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/solvers/stokes_2d/bc.py 
Description: File contains functions to reapply boundary conditions on the grids

Author: Marcel Ferrari, Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


import numba as nb
import importlib.metadata

# ======== Utilities for boundary conditions ========
# JIT-compiled to be callable from other JIT-compiled functions.

@nb.njit(cache=True)
def cpu_apply_vx_BC(vx, BC):
    """
    Apply boundary conditions to the x-velocity field in-place.
    """
    # Top and bottom
    vx[0, :]  = -BC * vx[1, :]
    vx[-1, :] = -BC * vx[-2, :]
    # Left wall
    vx[:, 0]  = 0.0
    # Right + ghost
    vx[:, -2:] = 0.0
    return vx


@nb.njit(cache=True)
def cpu_apply_vy_BC(vy, BC):
    """
    Apply boundary conditions to the y-velocity field in-place.
    """
    # Left and right
    vy[:, 0]   = -BC * vy[:, 1]
    vy[:, -1]  = -BC * vy[:, -2]
    # Top wall
    vy[0, :]   = 0.0
    # Bottom + ghost
    vy[-2:, :] = 0.0
    return vy


@nb.njit(cache=True)
def apply_p_BC(p):
    """
    Apply boundary conditions to the pressure field in-place.
    """
    # Dirichlet zero on all boundaries
    p[0, :] = 0.0
    p[-1, :] = 0.0
    p[:, 0] = 0.0
    p[:, -1] = 0.0
    return p


@nb.njit(cache=True)
def apply_BC(p, vx, vy, BC):
    """
    Apply BCs to pressure and velocity in-place, returning all arrays.
    """
    p  = apply_p_BC(p)
    vx = cpu_apply_vx_BC(vx, BC)
    vy = cpu_apply_vy_BC(vy, BC)
    return p, vx, vy


NUMBA_CUDA_AVAIL = False


try:
    importlib.metadata.version("numba-cuda")
    NUMBA_CUDA_AVAIL = True
except importlib.metadata.PackageNotFoundError:
    pass


# If Numba Cuda is available, prepare the decorators for the gpu exports as well.
if NUMBA_CUDA_AVAIL:
    import numba.cuda as cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

    @cuda.jit
    def gpu_apply_vx_bc_kernel(vx: DeviceNDArray, BC: float, nx1: int, ny1: int):
        """
        Cuda Kernel to apply boundary condition on the vx array
        """
        j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        if i >= ny1 or j >= nx1:
            return
        if i == 0:
            vx[0, j] = -BC * vx[1, j]
        elif i == ny1 - 1:
            vx[ny1 - 1, j] = -BC * vx[ny1 - 2, j]
        elif j == 0:
            vx[i, 0] = 0.0
        elif j >= nx1 - 2:
            vx[i, j] = 0.0


    @cuda.jit
    def gpu_apply_vy_bc_kernel(vy: DeviceNDArray, BC: float, nx1: int, ny1: int):
        """
        Cuda Kernel to apply boundary condition on the vy array
        """
        j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        if i >= ny1 or j >= nx1:
            return
        if j == 0:
            vy[i, 0] = -BC * vy[i, 1]
        elif j == nx1 - 1:
            vy[i, nx1 - 1] = -BC * vy[i, nx1 - 2]
        elif i == 0:
            vy[0, j] = 0.0
        elif i >= ny1 - 2:
            vy[i, j] = 0.0

else:
    def gpu_apply_vx_bc_kernel(*args, **kwargs):
        raise ImportError("numba-cuda needed for gpu support. Try installing Pyroclast[cuda]")

    def gpu_apply_vy_bc_kernel(*args, **kwargs):
        raise ImportError("numba-cuda needed for gpu support. Try installing Pyroclast[cuda]")
