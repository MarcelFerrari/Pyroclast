"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: implicit_operators.py
Description: This file implements the implicit operators for the Stokes flow
             and continuity equations in 2D.
Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import cupy as cp
from numba import cuda
import numpy as np

# Import device point kernels (already defined elsewhere)
# They must be declared as: @cuda.jit(device=True, inline="always")
from ..implicit_operators_stencils import vx_op_point, vy_op_point, p_op_point, \
                                        uzawa_vx_op_point, uzawa_vy_op_point, \
                                        uzawa_vx_rhs_point, uzawa_vy_rhs_point, \
                                        p_energy_point, vx_energy_point, vy_energy_point

# ========= JIT compile point kernels for GPU =========
vx_op_point_device = cuda.jit(inline="always")(vx_op_point)
vy_op_point_device = cuda.jit(inline="always")(vy_op_point)
p_op_point_device  = cuda.jit(inline="always")(p_op_point)
uzawa_vx_op_point_device = cuda.jit(inline="always")(uzawa_vx_op_point)
uzawa_vy_op_point_device = cuda.jit(inline="always")(uzawa_vy_op_point)
uzawa_vx_rhs_point_device = cuda.jit(inline="always")(uzawa_vx_rhs_point)
uzawa_vy_rhs_point_device = cuda.jit(inline="always")(uzawa_vy_rhs_point)
p_energy_point_device = cuda.jit(inline="always")(p_energy_point)
vx_energy_point_device = cuda.jit(inline="always")(vx_energy_point)
vy_energy_point_device = cuda.jit(inline="always")(vy_energy_point)

# ---------------------------
# Helpers
# ---------------------------

def _grid_block_from_array(arr, block=(16, 16)):
    """Return (grid, block) for a 2D array.

    The first element of ``block`` corresponds to the x-dimension (columns)
    and the second to the y-dimension (rows), matching CUDA's convention for
    ``blockDim``/``gridDim``.  Using this helper everywhere keeps the launch
    configuration consistent between kernels.
    """
    ny, nx = arr.shape
    block_x, block_y = block
    grid_x = (nx + block_x - 1) // block_x
    grid_y = (ny + block_y - 1) // block_y
    return (grid_x, grid_y), (block_x, block_y)

@cuda.jit
def _vx_operator_kernel(dx, dy, etap, etab, vx, vy, p, out, nx1, ny1):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # column (x)
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y  # row (y)
    # interior: i in [1, ny1-2], j in [1, nx1-3]
    if 1 <= i < ny1 - 1 and 1 <= j < nx1 - 2:
        out[i, j] = vx_op_point_device(i, j, dx, dy, etap, etab, vx, vy, p)

@cuda.jit
def _vy_operator_kernel(dx, dy, etap, etab, vx, vy, p, out, nx1, ny1):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # interior: i in [1, ny1-3], j in [1, nx1-2]
    if 1 <= i < ny1 - 2 and 1 <= j < nx1 - 1:
        out[i, j] = vy_op_point_device(i, j, dx, dy, etap, etab, vx, vy, p)

@cuda.jit
def _p_operator_kernel(dx, dy, vx, vy, out, nx1, ny1):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # interior: i in [1, ny1-2], j in [1, nx1-2]
    if 1 <= i < ny1 - 1 and 1 <= j < nx1 - 1:
        out[i, j] = p_op_point_device(i, j, dx, dy, vx, vy)

@cuda.jit
def _vx_sub_residual_kernel(rhs, op, out, nx1, ny1):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # vx interior: i ∈ [1, ny1-2], j ∈ [1, nx1-3]
    if 1 <= i < ny1 - 1 and 1 <= j < nx1 - 2:
        out[i, j] = rhs[i, j] - op[i, j]

@cuda.jit
def _vy_sub_residual_kernel(rhs, op, out, nx1, ny1):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # vy interior: i ∈ [1, ny1-3], j ∈ [1, nx1-2]
    if 1 <= i < ny1 - 2 and 1 <= j < nx1 - 1:
        out[i, j] = rhs[i, j] - op[i, j]

@cuda.jit
def _p_sub_residual_kernel(rhs, op, out, nx1, ny1):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # p interior: i ∈ [1, ny1-2], j ∈ [1, nx1-2]
    if 1 <= i < ny1 - 1 and 1 <= j < nx1 - 1:
        out[i, j] = rhs[i, j] - op[i, j]

@cuda.jit
def _uzawa_vx_rhs_kernel(dx, vx_rhs, p, out, nx1, ny1):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # i ∈ [1, ny1-2], j ∈ [1, nx1-3] (same as vx_operator interior)
    if 1 <= i < ny1 - 1 and 1 <= j < nx1 - 2:
        out[i, j] = uzawa_vx_rhs_point_device(i, j, dx, vx_rhs, p)

@cuda.jit
def _uzawa_vy_rhs_kernel(dy, vy_rhs, p, out, nx1, ny1):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # i ∈ [1, ny1-3], j ∈ [1, nx1-2] (same as vy_operator interior)
    if 1 <= i < ny1 - 2 and 1 <= j < nx1 - 1:
        out[i, j] = uzawa_vy_rhs_point_device(i, j, dy, vy_rhs, p)

@cuda.jit
def _uzawa_vx_operator_kernel(dx, dy, etap, etab, vx, vy, out, nx1, ny1):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if 1 <= i < ny1 - 1 and 1 <= j < nx1 - 2:
        out[i, j] = uzawa_vx_op_point_device(i, j, dx, dy, etap, etab, vx, vy)

@cuda.jit
def _uzawa_vy_operator_kernel(dx, dy, etap, etab, vx, vy, out, nx1, ny1):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if 1 <= i < ny1 - 2 and 1 <= j < nx1 - 1:
        out[i, j] = uzawa_vy_op_point_device(i, j, dx, dy, etap, etab, vx, vy)

@cuda.jit
def _p_energy_inplace_kernel(dx, dy, etap, p_res, nx1, ny1):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if 1 <= i < ny1 - 1 and 1 <= j < nx1 - 1:
        p_res[i, j] = p_energy_point_device(i, j, etap, p_res)

@cuda.jit
def _vx_energy_inplace_kernel(dx, dy, etap, etab, vx_res, nx1, ny1):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if 1 <= i < ny1 - 1 and 1 <= j < nx1 - 2:
        vx_res[i, j] = vx_energy_point_device(i, j, dx, dy, etap, etab, vx_res)

@cuda.jit
def _vy_energy_inplace_kernel(dx, dy, etap, etab, vy_res, nx1, ny1):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if 1 <= i < ny1 - 2 and 1 <= j < nx1 - 1:
        vy_res[i, j] = vy_energy_point_device(i, j, dx, dy, etap, etab, vy_res)

# ---------------------------
# Public GPU API (mirrors CPU API)
# ---------------------------

def vx_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, p, out):
    grid, block = _grid_block_from_array(out)
    _vx_operator_kernel[grid, block](dx, dy, etap, etab, vx, vy, p, out, nx1, ny1)
    cuda.synchronize()
    return out

def vx_residual(nx1, ny1, dx, dy, etap, etab, vx, vy, p, res_vx, rhs):
    vx_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, p, res_vx)
    grid, block = _grid_block_from_array(res_vx)
    _vx_sub_residual_kernel[grid, block](rhs, res_vx, res_vx, nx1, ny1)
    cuda.synchronize()
    return res_vx

def vy_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, p, out):
    grid, block = _grid_block_from_array(out)
    _vy_operator_kernel[grid, block](dx, dy, etap, etab, vx, vy, p, out, nx1, ny1)
    cuda.synchronize()
    return out

def vy_residual(nx1, ny1, dx, dy, etap, etab, vx, vy, p, res_vy, rhs):
    vy_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, p, res_vy)
    grid, block = _grid_block_from_array(res_vy)
    _vy_sub_residual_kernel[grid, block](rhs, res_vy, res_vy, nx1, ny1)
    cuda.synchronize()
    return res_vy

def p_operator(nx1, ny1, dx, dy, vx, vy, out):
    grid, block = _grid_block_from_array(out)
    _p_operator_kernel[grid, block](dx, dy, vx, vy, out, nx1, ny1)
    cuda.synchronize()
    return out

def p_residual(nx1, ny1, dx, dy, vx, vy, res_p, rhs):
    p_operator(nx1, ny1, dx, dy, vx, vy, res_p)
    grid, block = _grid_block_from_array(res_p)
    _p_sub_residual_kernel[grid, block](rhs, res_p, res_p, nx1, ny1)
    cuda.synchronize()
    return res_p

def uzawa_velocity_rhs(nx1, ny1, dx, dy, vx_rhs, vy_rhs, p, out_vx, out_vy):
    grid_vx, block_vx = _grid_block_from_array(out_vx)
    _uzawa_vx_rhs_kernel[grid_vx, block_vx](dx, vx_rhs, p, out_vx, nx1, ny1)
    cuda.synchronize()
    grid_vy, block_vy = _grid_block_from_array(out_vy)
    _uzawa_vy_rhs_kernel[grid_vy, block_vy](dy, vy_rhs, p, out_vy, nx1, ny1)
    cuda.synchronize()
    return out_vx, out_vy

def uzawa_vx_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, out):
    grid, block = _grid_block_from_array(out)
    _uzawa_vx_operator_kernel[grid, block](dx, dy, etap, etab, vx, vy, out, nx1, ny1)
    cuda.synchronize()
    return out

def uzawa_vx_residual(nx1, ny1, dx, dy, etap, etab, vx, vy, res_vx, rhs):
    uzawa_vx_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, res_vx)
    grid, block = _grid_block_from_array(res_vx)
    _vx_sub_residual_kernel[grid, block](rhs, res_vx, res_vx, nx1, ny1)
    cuda.synchronize()
    return res_vx

def uzawa_vy_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, out):
    grid, block = _grid_block_from_array(out)
    _uzawa_vy_operator_kernel[grid, block](dx, dy, etap, etab, vx, vy, out, nx1, ny1)
    cuda.synchronize()
    return out

def uzawa_vy_residual(nx1, ny1, dx, dy, etap, etab, vx, vy, res_vy, rhs):
    uzawa_vy_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, res_vy)
    grid, block = _grid_block_from_array(res_vy)
    _vy_sub_residual_kernel[grid, block](rhs, res_vy, res_vy, nx1, ny1)
    cuda.synchronize()
    return res_vy

def compute_p_energy_norm(nx1, ny1, dx, dy, etap, p_res):
    grid, block = _grid_block_from_array(p_res)
    _p_energy_inplace_kernel[grid, block](dx, dy, etap, p_res, nx1, ny1)
    cuda.synchronize()
    total = cp.sum(p_res[1:-1, 1:-1]).item()
    return np.sqrt(total)

def compute_vx_energy_norm(nx1, ny1, dx, dy, etap, etab, vx_res):
    grid, block = _grid_block_from_array(vx_res)
    _vx_energy_inplace_kernel[grid, block](dx, dy, etap, etab, vx_res, nx1, ny1)
    cuda.synchronize()
    total = cp.sum(vx_res[1:-1, 1:-2]).item()
    return np.sqrt(total)

def compute_vy_energy_norm(nx1, ny1, dx, dy, etap, etab, vy_res):
    grid, block = _grid_block_from_array(vy_res)
    _vy_energy_inplace_kernel[grid, block](dx, dy, etap, etab, vy_res, nx1, ny1)
    cuda.synchronize()
    total = cp.sum(vy_res[1:-2, 1:-1]).item()
    return np.sqrt(total)