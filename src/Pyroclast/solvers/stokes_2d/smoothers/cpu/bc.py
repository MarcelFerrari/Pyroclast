"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: solvers/stokes_2d/smoothers/cpu/bc.py
Description: CPU boundary condition helpers for Stokes smoother fields.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numba as nb

# ======== Utilities for boundary conditions ========
# JIT-compiled to be callable from other JIT-compiled functions.

@nb.njit(cache=True)
def apply_vx_BC(vx, BC):
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
def apply_vy_BC(vy, BC):
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
    vx = apply_vx_BC(vx, BC)
    vy = apply_vy_BC(vy, BC)
    return p, vx, vy
