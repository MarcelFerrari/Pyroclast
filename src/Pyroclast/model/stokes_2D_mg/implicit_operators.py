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

import numpy as np
import numba as nb

@nb.njit(cache=True, parallel=True)
def vx_operator(nx1, ny1,
                dx, dy,
                etap, etab,
                vx, vy, p, rax):
    """
    Compute the x-momentum residual for each interior cell.
    Returns res_x, a 2D array of the same shape as vx.
    """
    # Loop over interior
    for i in nb.prange(1, ny1 - 1):
        for j in nb.prange(1, nx1 - 2):
            # 1) Local viscosities
            etaA = etap[i,   j]
            etaB = etap[i,   j+1]
            eta1 = etab[i-1, j]
            eta2 = etab[i,   j]

            # 2) Coefficients for x-momentum operator
            vx1_coeff = 2.0 * etaA / (dx * dx)
            vx2_coeff = eta1 / (dy * dy)
            vx3_coeff = -(eta1 + eta2)/(dy * dy) - 2.0*(etaA + etaB)/(dx * dx)
            vx4_coeff = eta2 / (dy * dy)
            vx5_coeff = 2.0 * etaB / (dx * dx)

            # Cross terms with vy
            vy1_coeff =  eta1 / (dx * dy)
            vy2_coeff = -eta2 / (dx * dy)
            vy3_coeff = -eta1 / (dx * dy)
            vy4_coeff =  eta2 / (dx * dy)

            # Pressure gradient terms
            dp_right = -1.0/dx * p[i, j+1]
            dp_left  = +1.0/dx * p[i, j]

            # 3) Sum of off-diagonal contributions
            x_mom = (
                vx1_coeff * vx[i,   j-1] +
                vx2_coeff * vx[i-1, j  ] +
                vx3_coeff * vx[i,   j  ] +
                vx4_coeff * vx[i+1, j  ] +
                vx5_coeff * vx[i,   j+1]
                +
                vy1_coeff * vy[i-1, j  ] +
                vy2_coeff * vy[i,   j  ] +
                vy3_coeff * vy[i-1, j+1] +
                vy4_coeff * vy[i,   j+1]
                +
                dp_right + dp_left
            )
            
            rax[i, j] = x_mom
    return rax

@nb.njit(cache=True, parallel=True)
def vx_residual(nx1, ny1,
                dx, dy,
                etap, etab,
                vx, vy, p, res_vx, rhs):
    """
    Compute the x-momentum residual for each interior cell.
    Returns res_x, a 2D array of the same shape as vx.
    """
    
    # Store operator result in res_vx
    res_vx = vx_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, p, res_vx)

    # Compute residual
    for i in nb.prange(1, ny1 - 1):
        for j in nb.prange(1, nx1 - 2):
            res_vx[i, j] = rhs[i, j] - res_vx[i, j]
    
    return res_vx

@nb.njit(cache=True, parallel=True)
def vy_operator(nx1, ny1,
                dx, dy,
                etap, etab,
                vx, vy, p, rax):
    """
    Compute the y-momentum residual for each interior cell.
    Returns res_y, a 2D array of the same shape as vy.
    """
    # Loop over interior
    for i in nb.prange(1, ny1 - 2):
        for j in nb.prange(1, nx1 - 1):

            # 1) Local viscosities
            etaA = etap[i,   j]      # top cell's viscosity
            etaB = etap[i+1, j]      # bottom cell's viscosity
            eta1 = etab[i,   j-1]    
            eta2 = etab[i,   j]

            # 2) Coefficients for y-momentum operator
            vy1_coeff = eta1 / (dx * dx)
            vy2_coeff = 2.0 * etaA / (dy * dy)
            vy3_coeff = (-2.0 * etaA / (dy * dy)
                         -2.0 * etaB / (dy * dy)
                         -eta1 / (dx * dx)
                         -eta2 / (dx * dx))
            vy4_coeff = 2.0 * etaB / (dy * dy)
            vy5_coeff = eta2 / (dx * dx)

            # Cross terms with vx
            vx1_coeff =  eta1 / (dx * dy)
            vx2_coeff = -eta1 / (dx * dy)
            vx3_coeff = -eta2 / (dx * dy)
            vx4_coeff =  eta2 / (dx * dy)

            # Pressure gradient terms
            dp_up   = -1.0/dy * p[i+1, j]
            dp_down = +1.0/dy * p[i,   j]

            # 3) Sum of off-diagonal contributions
            y_mom = (
                vy1_coeff * vy[i,   j-1] +
                vy2_coeff * vy[i-1, j  ] +
                vy3_coeff * vy[i,   j  ] +
                vy4_coeff * vy[i+1, j  ] +
                vy5_coeff * vy[i,   j+1]
                +
                vx1_coeff * vx[i,   j-1] +
                vx2_coeff * vx[i+1, j-1] +
                vx3_coeff * vx[i,   j  ] +
                vx4_coeff * vx[i+1, j  ]
                +
                dp_up + dp_down
            )

            rax[i, j] = y_mom
    return rax

@nb.njit(cache=True, parallel=True)
def vy_residual(nx1, ny1,
                dx, dy,
                etap, etab,
                vx, vy, p, res_vy, rhs):
    """
    Compute the y-momentum residual for each interior cell.
    Returns res_y, a 2D array of the same shape as vy.
    """
    # Store operator result in res_vy
    res_vy = vy_operator(nx1, ny1,
                         dx, dy,
                         etap, etab,
                         vx, vy, p, res_vy)

    # Compute residual
    for i in nb.prange(1, ny1 - 2):
        for j in nb.prange(1, nx1 - 1):
            res_vy[i, j] = rhs[i, j] - res_vy[i, j]
    
    return res_vy

@nb.njit(cache=True, parallel=True)
def p_operator(nx1, ny1,
               dx, dy,
               vx, vy,
               rax):
    """
    Compute the continuity (pressure) residual for each cell.
    Returns res_p, a 2D array of the same shape as p.
    """
    # For convenience, we only compute on the interior
    for i in nb.prange(1, ny1 - 1):
        for j in nb.prange(1, nx1 - 1):
            rax[i, j] = ((vx[i, j] - vx[i, j - 1]) / dx +
                        (vy[i, j] - vy[i - 1, j]) / dy)
    return rax
            
@nb.njit(cache=True, parallel=True)
def p_residual(nx1, ny1,
               dx, dy,
               vx, vy,
               res_p, rhs):
    """
    Compute the continuity (pressure) residual for each cell.
    Returns res_p, a 2D array of the same shape as p.
    """
    # Store operator result in res_p
    res_p = p_operator(nx1, ny1, dx, dy, vx, vy, res_p)

    # Compute residual
    for i in nb.prange(1, ny1 - 1):
        for j in nb.prange(1, nx1 - 1):
            res_p[i, j] = rhs[i, j] - res_p[i, j]
    
    return res_p
