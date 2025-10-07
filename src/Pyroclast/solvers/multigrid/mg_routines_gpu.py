"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/solvers/multigrid/mg_routines_gpu.py 
Description: File contains kernels and routines for dealing with multi-grid levels on GPU

Author: Marcel Ferrari, Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


# TODO stubs?


import cupy as cp
from numba import cuda

# -----------------------
# Device helpers
# -----------------------

@cuda.jit(device=True, inline=True)
def _clip_i(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def _grid2d(ny, nx, block=(16, 16)):
    by, bx = block
    gy = (ny + by - 1) // by
    gx = (nx + bx - 1) // bx
    return (gx, gy), (bx, by)
# -----------------------
# Kernels
# -----------------------

@cuda.jit
def _restrict2d_scatter(
    nxh, nyh, xh, yh, uh,
    nxH, nyH, xH, yH,
    uH_accum, wH_accum,
    dxH, dyH, xH0, yH0
):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i >= nyh - 1 or j >= nxh - 1:
        return

    yhi = yh[i]
    xhj = xh[j]

    iH = int((yhi - yH0) / dyH)
    jH = int((xhj - xH0) / dxH)
    iH = _clip_i(iH, 0, nyH - 2)
    jH = _clip_i(jH, 0, nxH - 2)

    ry = (yhi - yH[iH]) / dyH
    rx = (xhj - xH[jH]) / dxH

    w00 = (1.0 - rx) * (1.0 - ry)
    w01 = rx * (1.0 - ry)
    w10 = (1.0 - rx) * ry
    w11 = rx * ry

    v = uh[i, j]

    cuda.atomic.add(uH_accum, (iH,   jH  ), w00 * v)
    cuda.atomic.add(uH_accum, (iH+1, jH  ), w10 * v)
    cuda.atomic.add(uH_accum, (iH,   jH+1), w01 * v)
    cuda.atomic.add(uH_accum, (iH+1, jH+1), w11 * v)

    cuda.atomic.add(wH_accum, (iH,   jH  ), w00)
    cuda.atomic.add(wH_accum, (iH+1, jH  ), w10)
    cuda.atomic.add(wH_accum, (iH,   jH+1), w01)
    cuda.atomic.add(wH_accum, (iH+1, jH+1), w11)

@cuda.jit
def _normalize2d(uH, wH):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    ny, nx = uH.shape
    if i >= ny or j >= nx:
        return
    w = wH[i, j]
    if w > 0.0:
        uH[i, j] = uH[i, j] / w

@cuda.jit
def _prolong2d(
    nxH, nyH, xH, yH, uH,
    nxh, nyh, xh, yh, uh,
    dxH, dyH, xH0, yH0
):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i >= nyh or j >= nxh:
        return

    yhi = yh[i]
    xhj = xh[j]

    iH = int((yhi - yH0) / dyH)
    jH = int((xhj - xH0) / dxH)
    iH = _clip_i(iH, 0, nyH - 2)
    jH = _clip_i(jH, 0, nxH - 2)

    ry = (yhi - yH[iH]) / dyH
    rx = (xhj - xH[jH]) / dxH

    uh[i, j] = (1.0 - rx) * (1.0 - ry) * uH[iH,   jH  ] + \
               (      rx) * (1.0 - ry) * uH[iH,   jH+1] + \
               (1.0 - rx) * (      ry) * uH[iH+1, jH  ] + \
               (      rx) * (      ry) * uH[iH+1, jH+1]

# -----------------------
# Public API (same signatures)
# -----------------------

def restrict_2D(nxh, nyh, xh, yh, uh, nxH, nyH, xH, yH, uH, uHw):
    """
    In-place GPU restriction (fine->coarse) using atomics.
    All arrays are CuPy and preallocated. Returns uH (normalized).
    """
    # zero accumulators (in-place, no allocation)
    uH.fill(0)
    uHw.fill(0)

    # scalars (host)
    dxH = float((xH[1] - xH[0]).item())
    dyH = float((yH[1] - yH[0]).item())
    xH0 = float(xH[0].item())
    yH0 = float(yH[0].item())

    # launch over fine interior (exclude last row/col)
    by, bx = 16, 16
    gy = (nyh - 1 + by - 1) // by
    gx = (nxh - 1 + bx - 1) // bx
    _restrict2d_scatter[(gx, gy), (bx, by)](
        nxh, nyh, xh, yh, uh,
        nxH, nyH, xH, yH,
        uH, uHw,
        dxH, dyH, xH0, yH0
    )

    # normalize uH by weights uHw
    gyN = (nyH + by - 1) // by
    gxN = (nxH + bx - 1) // bx
    _normalize2d[(gxN, gyN), (bx, by)](uH, uHw)

    return uH

def prolong_2D(nxH, nyH, xH, yH, uH,
               nxh, nyh, xh, yh, uh):
    """
    In-place GPU prolongation (coarse->fine).
    All arrays are CuPy and preallocated. Returns uh.
    """
    dxH = float((xH[1] - xH[0]).item())
    dyH = float((yH[1] - yH[0]).item())
    xH0 = float(xH[0].item())
    yH0 = float(yH[0].item())

    by, bx = 16, 16
    gy = (nyh + by - 1) // by
    gx = (nxh + bx - 1) // bx
    _prolong2d[(gx, gy), (bx, by)](
        nxH, nyH, xH, yH, uH,
        nxh, nyh, xh, yh, uh,
        dxH, dyH, xH0, yH0
    )
    return uh
