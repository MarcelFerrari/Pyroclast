"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: solvers/stokes_2d/smoothers/gpu/jacobi.py
Description: CUDA Jacobi smoothing kernels and launch wrappers for Stokes solves.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from Pyroclast.gpu_utils import launch_2D, get_numba_stream
import numba as nb
from numba import cuda
import cupy as cp

from .bc import apply_p_BC, apply_vx_BC, apply_vy_BC
from ..coeff import x_momentum_coefficients as _x_coeff,\
                    y_momentum_coefficients as _y_coeff

# Device-callable wrappers for your coefficient functions
x_momentum_coefficients = cuda.jit(inline="always")(_x_coeff)
y_momentum_coefficients = cuda.jit(inline="always")(_y_coeff)


# =========================
#   CUDA kernels
# =========================
@cuda.jit
def _pressure_sweep_interior_kernel(nx1, ny1, dx, dy,
                                    vx, vy, p, beta, relax_p, rhs):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if 1 <= i < ny1 - 1 and 1 <= j < nx1 - 1:
        res = rhs[i, j] - ((vx[i, j] - vx[i, j - 1]) / dx +
                           (vy[i, j] - vy[i - 1, j]) / dy)
        p[i, j] += res * beta[i, j] * relax_p


@cuda.jit
def _vx_jacobi_sweep_kernel(nx1, ny1, dx, dy,
                            etap, etab,
                            vx_old, vy_old,
                            relax_v, rhs,
                            vx_new_out):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # interior: i in [1..ny1-2], j in [1..nx1-3]
    if 1 <= i < ny1 - 1 and 1 <= j < nx1 - 2:
        etaA = etap[i,   j]
        etaB = etap[i,   j+1]
        eta1 = etab[i-1, j]
        eta2 = etab[i,   j]

        vx1, vx2, vx3, vx4, vx5, vy1, vy2, vy3, vy4 = x_momentum_coefficients(
            dx, dy, etaA, etaB, eta1, eta2
        )

        sum_neighbors = (
            vx1 * vx_old[i,   j-1] +
            vx2 * vx_old[i-1, j  ] +
            vx4 * vx_old[i+1, j  ] +
            vx5 * vx_old[i,   j+1]
            +
            vy1 * vy_old[i-1, j  ] +
            vy2 * vy_old[i,   j  ] +
            vy3 * vy_old[i-1, j+1] +
            vy4 * vy_old[i,   j+1]
        )

        vx_new_out[i, j] = (1.0 - relax_v) * vx_old[i, j] + relax_v * (rhs[i, j] - sum_neighbors) / vx3


@cuda.jit
def _vy_jacobi_sweep_kernel(nx1, ny1, dx, dy,
                            etap, etab,
                            vx_old, vy_old,
                            relax_v, rhs,
                            vy_new_out):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # interior: i in [1..ny1-3], j in [1..nx1-2]
    if 1 <= i < ny1 - 2 and 1 <= j < nx1 - 1:
        etaA = etap[i,   j]
        etaB = etap[i+1, j]
        eta1 = etab[i,   j-1]
        eta2 = etab[i,   j]

        vy1, vy2, vy3, vy4, vy5, vx1, vx2, vx3, vx4 = y_momentum_coefficients(
            dx, dy, etaA, etaB, eta1, eta2
        )

        sum_neighbors = (
            vy1 * vy_old[i,   j-1] +
            vy2 * vy_old[i-1, j  ] +
            vy4 * vy_old[i+1, j  ] +
            vy5 * vy_old[i,   j+1]
            +
            vx1 * vx_old[i,   j-1] +
            vx2 * vx_old[i+1, j-1] +
            vx3 * vx_old[i,   j  ] +
            vx4 * vx_old[i+1, j  ]
        )

        vy_new_out[i, j] = (1.0 - relax_v) * vy_old[i, j] + relax_v * (rhs[i, j] - sum_neighbors) / vy3


# =========================
#   Public API (unchanged)
# =========================
def pressure_sweep(nx1, ny1, dx, dy,
                   vx, vy, p,
                   beta,
                   relax_p, rhs):
    # 1) interior update on GPU
    grid, block = launch_2D((ny1, nx1))
    stream = get_numba_stream()
    _pressure_sweep_interior_kernel[grid, block, stream](nx1, ny1, dx, dy, vx, vy, p, beta, relax_p, rhs)

    # 2) zero-mean using CuPy reductions (interior mean)
    pbar = cp.mean(p[1:ny1-1, 1:nx1-1])
    p -= pbar  # in-place device op

    # 3) pressure BCs (provided by your codebase)
    apply_p_BC(p)
    return p

def jacobi_velocity_smoother(nx1, ny1,
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
    grid, block = launch_2D((ny1, nx1))
    stream = get_numba_stream()

    for _ in range(max_iter):
        # vx sweep
        _vx_jacobi_sweep_kernel[grid, block, stream](nx1, ny1, dx, dy, etap, etab,
                                             vx, vy, relax_v, vx_rhs, vx_new)
        apply_vx_BC(vx_new, BC)

        # vy sweep
        _vy_jacobi_sweep_kernel[grid, block, stream](nx1, ny1, dx, dy, etap, etab,
                                             vx, vy, relax_v, vy_rhs, vy_new)
        apply_vy_BC(vy_new, BC)

        # ping-pong
        vx, vx_new = vx_new, vx
        vy, vy_new = vy_new, vy

    return vx, vy
