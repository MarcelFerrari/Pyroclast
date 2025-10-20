"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: solvers/stokes_2d/smoothers/gpu/jacobi_fused.py
Description: Fused GPU Jacobi smoothers for coupled Stokes velocity-pressure updates.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from numba import cuda, types
import cupy as cp
import Pyroclast.gpu_utils as gpu_utils

# ======== Utilities for boundary conditions ========
@cuda.jit(inline=True, fastmath=True, cache=True)
def _apply_vx_BC_device(vx, BC, nx1, ny1):
    # Thread’s “home” lane and strides
    j0 = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i0 = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    j_stride = cuda.blockDim.x * cuda.gridDim.x
    i_stride = cuda.blockDim.y * cuda.gridDim.y

    # --- Top row: i == 0
    for j in range(j0, nx1, j_stride):
        vx[0, j] = -BC * vx[1, j]

    # --- Bottom row: i == ny1-1
    for j in range(j0, nx1, j_stride):
        vx[ny1 - 1, j] = -BC * vx[ny1 - 2, j]

    # --- Left column: j == 0  (skip corners to avoid double write)
    for i in range(1 + i0, ny1 - 1, i_stride):
        vx[i, 0] = 0.0

    # --- Right columns: j in {nx1-2, nx1-1} (skip corners)
    # mirror your original `j >= nx1 - 2`
    j_right0 = nx1 - 2
    if j_right0 >= 0:
        for i in range(1 + i0, ny1 - 1, i_stride):
            # j = nx1-2
            if j_right0 < nx1:
                vx[i, j_right0] = 0.0
            # j = nx1-1
            if j_right0 + 1 < nx1:
                vx[i, j_right0 + 1] = 0.0


@cuda.jit(inline=True, fastmath=True, cache=True,)
def _apply_vy_BC_device(vy, BC, nx1, ny1):
    j0 = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i0 = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    j_stride = cuda.blockDim.x * cuda.gridDim.x
    i_stride = cuda.blockDim.y * cuda.gridDim.y

    # --- Left column: j == 0
    for i in range(i0, ny1, i_stride):
        vy[i, 0] = -BC * vy[i, 1]

    # --- Right column: j == nx1-1
    jR = nx1 - 1
    if jR >= 0:
        for i in range(i0, ny1, i_stride):
            vy[i, jR] = -BC * vy[i, jR - 1]

    # --- Top row: i == 0  (skip corners)
    for j in range(1 + j0, nx1 - 1, j_stride):
        vy[0, j] = 0.0

    # --- Bottom rows: i in {ny1-2, ny1-1} (skip corners)
    i_bottom0 = ny1 - 2
    if i_bottom0 >= 0:
        for j in range(1 + j0, nx1 - 1, j_stride):
            # i = ny1-2
            if i_bottom0 < ny1:
                vy[i_bottom0, j] = 0.0
            # i = ny1-1
            if i_bottom0 + 1 < ny1:
                vy[i_bottom0 + 1, j] = 0.0


from ..coeff import x_momentum_coefficients as _x_coeff,\
                    y_momentum_coefficients as _y_coeff

# Device-callable wrappers for your coefficient functions
x_momentum_coefficients = cuda.jit(inline=True, cache=True)(_x_coeff)
y_momentum_coefficients = cuda.jit(inline=True, cache=True)(_y_coeff)
    
# =========================
#   CUDA kernels
# =========================
@cuda.jit(fastmath=True, cache=True)
def _pressure_sweep_interior_kernel(nx1, ny1, dx, dy,
                                    vx, vy, p, beta, relax_p, rhs):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if 1 <= i < ny1 - 1 and 1 <= j < nx1 - 1:
        res = rhs[i, j] - ((vx[i, j] - vx[i, j - 1]) / dx +
                           (vy[i, j] - vy[i - 1, j]) / dy)
        p[i, j] += res * beta[i, j] * relax_p


@cuda.jit(inline=True, fastmath=True, cache=True)
def _vx_jacobi_sweep_kernel(nx1, ny1, dx, dy,
                            etap, etab,
                            vx, vy,
                            relax_v, rhs,
                            vx_new):
    # Thread’s starting indices
    j0 = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i0 = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Strides the thread will hop by to cover the whole domain
    j_stride = cuda.blockDim.x * cuda.gridDim.x
    i_stride = cuda.blockDim.y * cuda.gridDim.y

    # Interior for vx: i in [1..ny1-2], j in [1..nx1-3]
    # range(stop) is exclusive, so stops are ny1-1 and nx1-2.
    start_i = 1 + i0
    start_j = 1 + j0

    for i in range(start_i, ny1 - 1, i_stride):
        for j in range(start_j, nx1 - 2, j_stride):
            etaA = etap[i,   j]
            etaB = etap[i,   j+1]
            eta1 = etab[i-1, j]
            eta2 = etab[i,   j]

            vx1, vx2, vx3, vx4, vx5, vy1, vy2, vy3, vy4 = x_momentum_coefficients(
                dx, dy, etaA, etaB, eta1, eta2
            )

            sum_neighbors = (
                vx1 * vx[i,   j-1] +
                vx2 * vx[i-1, j  ] +
                vx4 * vx[i+1, j  ] +
                vx5 * vx[i,   j+1] +
                vy1 * vy[i-1, j  ] +
                vy2 * vy[i,   j  ] +
                vy3 * vy[i-1, j+1] +
                vy4 * vy[i,   j+1]
            )

            vx_new[i, j] = (1.0 - relax_v) * vx[i, j] + relax_v * (rhs[i, j] - sum_neighbors) / vx3



@cuda.jit(inline=True, fastmath=True, cache=True)
def _vy_jacobi_sweep_kernel(nx1, ny1, dx, dy,
                            etap, etab,
                            vx, vy,
                            relax_v, rhs,
                            vy_new):
    j0 = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i0 = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    j_stride = cuda.blockDim.x * cuda.gridDim.x
    i_stride = cuda.blockDim.y * cuda.gridDim.y

    # Interior for vy: i in [1..ny1-3], j in [1..nx1-2]
    # So stops are ny1-2 and nx1-1.
    start_i = 1 + i0
    start_j = 1 + j0

    for i in range(start_i, ny1 - 2, i_stride):
        for j in range(start_j, nx1 - 1, j_stride):
            etaA = etap[i,   j]
            etaB = etap[i+1, j]
            eta1 = etab[i,   j-1]
            eta2 = etab[i,   j]

            vy1, vy2, vy3, vy4, vy5, vx1, vx2, vx3, vx4 = y_momentum_coefficients(
                dx, dy, etaA, etaB, eta1, eta2
            )

            sum_neighbors = (
                vy1 * vy[i,   j-1] +
                vy2 * vy[i-1, j  ] +
                vy4 * vy[i+1, j  ] +
                vy5 * vy[i,   j+1] +
                vx1 * vx[i,   j-1] +
                vx2 * vx[i+1, j-1] +
                vx3 * vx[i,   j  ] +
                vx4 * vx[i+1, j  ]
            )

            vy_new[i, j] = (1.0 - relax_v) * vy[i, j] + relax_v * (rhs[i, j] - sum_neighbors) / vy3


# Need fixed signature for cooperative kernel launch
_SIG = (
    types.int32, types.int32,          # nx1, ny1
    types.float64, types.float64,      # dx, dy
    types.float64[:, ::1], types.float64[:, ::1],  # etap, etab
    types.float64[:, ::1], types.float64[:, ::1],  # vx, vy
    types.float64,                     # relax_v
    types.float64[:, ::1], types.float64[:, ::1],  # vx_rhs, vy_rhs
    types.float64[:, ::1], types.float64[:, ::1],  # vx_new, vy_new
    types.float64,                     # BC
    types.int32                        # max_iter
)

@cuda.jit(_SIG, fastmath=True, cache=True)
def _jacobi_fused_kernel(nx1, ny1, dx, dy,
                         etap, etab,
                         vx, vy,
                         relax_v, vx_rhs, vy_rhs,
                         vx_new, vy_new, BC, max_iter):
    g = cuda.cg.this_grid()
    for _ in range(max_iter):
        _vx_jacobi_sweep_kernel(nx1, ny1, dx, dy, etap, etab, vx, vy, relax_v, vx_rhs, vx_new)
        _vy_jacobi_sweep_kernel(nx1, ny1, dx, dy, etap, etab, vx, vy, relax_v, vy_rhs, vy_new)

        g.sync()

        _apply_vx_BC_device(vx_new, BC, nx1, ny1)
        _apply_vy_BC_device(vy_new, BC, nx1, ny1)

        g.sync()

        # ping-pong
        vx, vx_new = vx_new, vx
        vy, vy_new = vy_new, vy


# =========================
#   Public API
# =========================
def jacobi_velocity_smoother_fused(nx1, ny1,
                                   dx, dy,
                                   etap, etab,
                                   vx, vy,              # initial guesses (also serve as buffers)
                                   vx_new, vy_new,      # alternate preallocated buffers
                                   relax_v, BC,
                                   vx_rhs, vy_rhs,
                                   max_iter):
    
    # Check that arrays are already on GPU
    assert isinstance(etap, cp.ndarray)
    assert isinstance(etab, cp.ndarray)
    assert isinstance(vx, cp.ndarray)
    assert isinstance(vy, cp.ndarray)
    assert isinstance(vx_new, cp.ndarray)
    assert isinstance(vy_new, cp.ndarray)

    # keep even number of iterations
    max_iter += max_iter % 2
    
    blocksize = gpu_utils.get_block_size()
    kernel = _jacobi_fused_kernel.overloads[_SIG]
    max_blocks = kernel.max_cooperative_grid_blocks(blocksize)
    grid, block = gpu_utils.launch_2D((ny1, nx1), max_blocks=max_blocks)
    
    stream = gpu_utils.get_numba_stream()
    
    _jacobi_fused_kernel[grid, block, stream](nx1, ny1,
                                              dx, dy,
                                              etap, etab,
                                              vx, vy,
                                              relax_v, vx_rhs, vy_rhs,
                                              vx_new, vy_new, BC, max_iter)

   
    return vx, vy
