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

from ..implicit_operators_stencils import vx_op_point, vy_op_point, p_op_point, \
                                        uzawa_vx_op_point, uzawa_vy_op_point, \
                                        uzawa_vx_rhs_point, uzawa_vy_rhs_point, \
                                        p_energy_point, vx_energy_point, vy_energy_point
import numpy as np
import numba as nb

# ========= JIT compile point kernels for CPU =========
vx_op_point = nb.njit(vx_op_point, inline='always', cache=True)
vy_op_point = nb.njit(vy_op_point, inline='always', cache=True)
p_op_point  = nb.njit(p_op_point,  inline='always', cache=True)
uzawa_vx_op_point = nb.njit(uzawa_vx_op_point, inline='always', cache=True)
uzawa_vy_op_point = nb.njit(uzawa_vy_op_point, inline='always', cache=True)
uzawa_vx_rhs_point = nb.njit(uzawa_vx_rhs_point, inline='always', cache=True)
uzawa_vy_rhs_point = nb.njit(uzawa_vy_rhs_point, inline='always', cache=True)
p_energy_point = nb.njit(p_energy_point, inline='always', cache=True)
vx_energy_point = nb.njit(vx_energy_point, inline='always', cache=True)
vy_energy_point = nb.njit(vy_energy_point, inline='always', cache=True)

# ========= Operators =========
@nb.njit(cache=True, parallel=True)
def vx_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, p, out):
    for i in nb.prange(1, ny1 - 1):
        for j in range(1, nx1 - 2):
            out[i, j] = vx_op_point(i, j, dx, dy, etap, etab, vx, vy, p)
    return out


@nb.njit(cache=True, parallel=True)
def vx_residual(nx1, ny1, dx, dy, etap, etab, vx, vy, p, res_vx, rhs):
    res_vx = vx_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, p, res_vx)
    for i in nb.prange(1, ny1 - 1):
        for j in range(1, nx1 - 2):
            res_vx[i, j] = rhs[i, j] - res_vx[i, j]
    return res_vx

@nb.njit(cache=True, parallel=True)
def vy_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, p, out):
    for i in nb.prange(1, ny1 - 2):
        for j in range(1, nx1 - 1):
            out[i, j] = vy_op_point(i, j, dx, dy, etap, etab, vx, vy, p)
    return out

@nb.njit(cache=True, parallel=True)
def vy_residual(nx1, ny1, dx, dy, etap, etab, vx, vy, p, res_vy, rhs):
    res_vy = vy_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, p, res_vy)
    for i in nb.prange(1, ny1 - 2):
        for j in range(1, nx1 - 1):
            res_vy[i, j] = rhs[i, j] - res_vy[i, j]
    return res_vy

@nb.njit(cache=True, parallel=True)
def p_operator(nx1, ny1, dx, dy, vx, vy, out):
    for i in nb.prange(1, ny1 - 1):
        for j in range(1, nx1 - 1):
            out[i, j] = p_op_point(i, j, dx, dy, vx, vy)
    return out

@nb.njit(cache=True, parallel=True)
def p_residual(nx1, ny1, dx, dy, vx, vy, res_p, rhs):
    res_p = p_operator(nx1, ny1, dx, dy, vx, vy, res_p)
    for i in nb.prange(1, ny1 - 1):
        for j in range(1, nx1 - 1):
            res_p[i, j] = rhs[i, j] - res_p[i, j]
    return res_p

@nb.njit(cache=True, parallel=True)
def uzawa_velocity_rhs(nx1, ny1, dx, dy, vx_rhs, vy_rhs, p, out_vx, out_vy):
    # vx Uzawa RHS
    for i in nb.prange(1, ny1 - 1):
        for j in range(1, nx1 - 2):
            out_vx[i, j] = uzawa_vx_rhs_point(i, j, dx, vx_rhs, p)
    # vy Uzawa RHS
    for i in nb.prange(1, ny1 - 2):
        for j in range(1, nx1 - 1):
            out_vy[i, j] = uzawa_vy_rhs_point(i, j, dy, vy_rhs, p)
    return out_vx, out_vy

@nb.njit(cache=True, parallel=True)
def uzawa_vx_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, out):
    for i in nb.prange(1, ny1 - 1):
        for j in range(1, nx1 - 2):
            out[i, j] = uzawa_vx_op_point(i, j, dx, dy, etap, etab, vx, vy)
    return out

@nb.njit(cache=True, parallel=True)
def uzawa_vx_residual(nx1, ny1, dx, dy, etap, etab, vx, vy, res_vx, rhs):
    res_vx = uzawa_vx_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, res_vx)
    for i in nb.prange(1, ny1 - 1):
        for j in range(1, nx1 - 2):
            res_vx[i, j] = rhs[i, j] - res_vx[i, j]
    return res_vx

@nb.njit(cache=True, parallel=True)
def uzawa_vy_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, out):
    for i in nb.prange(1, ny1 - 2):
        for j in range(1, nx1 - 1):
            out[i, j] = uzawa_vy_op_point(i, j, dx, dy, etap, etab, vx, vy)
    return out

@nb.njit(cache=True, parallel=True)
def uzawa_vy_residual(nx1, ny1, dx, dy, etap, etab, vx, vy, res_vy, rhs):
    res_vy = uzawa_vy_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, res_vy)
    for i in nb.prange(1, ny1 - 2):
        for j in range(1, nx1 - 1):
            res_vy[i, j] = rhs[i, j] - res_vy[i, j]
    return res_vy

# ========= Energy norms (single scalar reduction inferred by Numba) =========

@nb.njit(cache=True, parallel=True)
def compute_p_energy_norm(nx1, ny1, dx, dy, etap, p_res):
    total = 0.0
    for j in nb.prange(1, ny1 - 1):
        for i in range(1, nx1 - 1):
            total += p_energy_point(i, j, etap, p_res)
    return np.sqrt(total)

@nb.njit(cache=True, parallel=True)
def compute_vx_energy_norm(nx1, ny1, dx, dy, etap, etab, vx_res):
    total = 0.0
    for i in nb.prange(1, ny1 - 1):
        for j in range(1, nx1 - 2):
            total += vx_energy_point(i, j, dx, dy, etap, etab, vx_res)
    return np.sqrt(total)

@nb.njit(cache=True, parallel=True)
def compute_vy_energy_norm(nx1, ny1, dx, dy, etap, etab, vy_res):
    total = 0.0
    for i in nb.prange(1, ny1 - 2):
        for j in range(1, nx1 - 1):
            total += vy_energy_point(i, j, dx, dy, etap, etab, vy_res)
    return np.sqrt(total)
