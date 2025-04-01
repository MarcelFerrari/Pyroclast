"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: utils.py
Description: This file implements utility functions for the multigrid solver.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numba as nb
import numpy as np

@nb.njit(cache=True)
def compute_hydrostatic_pressure(nx1, ny1, dy, rho, gy, p_ref):
    """
    Compute the hydrostatic pressure field.
    """
    p = np.zeros((ny1, nx1))
    
    # Set pressure at the top boundary
    for j in range(1, nx1-1):
        p[1, j] = p_ref

    # Compute pressure field
    for i in range(2, ny1-1):
        for j in range(1, nx1-1):
            p[i, j] = p[i-1, j] + gy * dy * (rho[i, j] + rho[i-1, j])/2

    return p


# ======== Utilities for boundary conditions ========
# These are JIT-compiled in order to make them callable from other JIT-compiled functions.
@nb.njit(cache=True)
def apply_vx_BC(vx, BC):
    """
    Apply boundary conditions to the x-velocity field.
    """
    # Apply vx boundary conditions
    vx[0, :]  = -BC * vx[1, :]    # Top
    vx[-1, :] = -BC * vx[-2, :]   # Bottom
    vx[:, 0]  = 0.0               # Left
    vx[:, -2:] = 0.0              # Right + ghost

@nb.njit(cache=True)
def apply_vy_BC(vy, BC):
    # Apply vy boundary conditions
    vy[:, 0]   = -BC * vy[:, 1]  # Left
    vy[:, -1]  = -BC * vy[:, -2] # Right
    vy[0, :]   = 0.0             # Top
    vy[-2:, :] = 0.0             # Bottom + ghost


@nb.njit(cache=True)
def apply_p_BC(p):
    # Fix pressure at the boundaries
    p[0, :] = 0.0   # Top
    p[-1, :] = 0.0  # Bottom
    p[:, 0] = 0.0   # Left
    p[:, -1] = 0.0  # Right


@nb.njit(cache=True)
def apply_BC(p, vx, vy, BC):
    """
    Apply boundary conditions to the velocity and pressure fields.
    """
    apply_p_BC(p)
    apply_vx_BC(vx, BC)
    apply_vy_BC(vy, BC)


@nb.njit(cache=True)
def apply_u_BC(u, BC):
    """
    Apply boundary conditions to the velocity and pressure fields.
    """
    # Apply vx boundary conditions
    u[0, :, 0]  = -BC * u[1, :, 0]    # Top
    u[-1, :, 0] = -BC * u[-2, :, 0]   # Bottom
    u[:, 0, 0]  = 0.0               # Left
    u[:, -2:, 0] = 0.0              # Right + ghost

    # Apply vy boundary conditions
    u[:, 0, 1]   = -BC * u[:, 1, 1]  # Left
    u[:, -1, 1]  = -BC * u[:, -2, 1] # Right
    u[0, :, 1]   = 0.0             # Top
    u[-2:, :, 1] = 0.0             # Bottom + ghost
    
    



