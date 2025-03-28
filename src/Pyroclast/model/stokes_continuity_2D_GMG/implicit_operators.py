"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: smoother.py
Description: This file implements jacobi and red-black Gauss-Seidel smoothers
             for the Stokes flow and continuity equations in 2D.
             The pressure update is done using the Uzawa method.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

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
    vx_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, p, res_vx)

    # Compute residual
    for i in nb.prange(1, ny1 - 1):
        for j in nb.prange(1, nx1 - 2):
            res_vx[i, j] = rhs[i, j] - res_vx[i, j]

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
    vy_operator(nx1, ny1,
                dx, dy,
                etap, etab,
                vx, vy, p, res_vy)

    # Compute residual
    for i in nb.prange(1, ny1 - 2):
        for j in nb.prange(1, nx1 - 1):
            res_vy[i, j] = rhs[i, j] - res_vy[i, j]


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
    p_operator(nx1, ny1, dx, dy, vx, vy, res_p)

    # Compute residual
    for i in nb.prange(1, ny1 - 1):
        for j in nb.prange(1, nx1 - 1):
            res_p[i, j] = rhs[i, j] - res_p[i, j]


# Assemble rhs of momentum equations assuming fixed pressure
# @nb.njit(cache = True, parallel = True)
# def assemble_momentum(nx1, ny1, dx, dy, rho, gy, p):
#     # Allocate rhs
#     b = np.zeros((2, ny1, nx1), dtype=np.float64)

#     for i in nb.prange(1, ny1 - 1):
#         for j in range(1, nx1 - 2):
#             b[0, i, j] = (p[i, j+1] - p[i, j])/dx

#     for i in nb.prange(1, ny1 - 2):
#         for j in range(1, nx1 - 1):
#             b[0, i, j] = -gy*rho[i, j] + (p[i+1, j] - p[i, j])/dy
    
#     return b
    
# @nb.njit(cache=True, parallel=True)
# def fixed_p_vx_operator(nx1, ny1,
#                         dx, dy,
#                         etap, etab,
#                         vx, vy, rax, BC):
#     """
#     Compute the x-momentum residual for each interior cell.
#     Returns res_x, a 2D array of the same shape as vx.
#     """
#     # Loop over interior
#     for i in nb.prange(1, ny1 - 1):
#         for j in nb.prange(1, nx1 - 2):
#             # 1) Local viscosities
#             etaA = etap[i,   j]
#             etaB = etap[i,   j+1]
#             eta1 = etab[i-1, j]
#             eta2 = etab[i,   j]

#             # 2) Coefficients for x-momentum operator
#             vx1_coeff = 2.0 * etaA / (dx * dx)
#             vx2_coeff = eta1 / (dy * dy)
#             vx3_coeff = -(eta1 + eta2)/(dy * dy) - 2.0*(etaA + etaB)/(dx * dx)
#             vx4_coeff = eta2 / (dy * dy)
#             vx5_coeff = 2.0 * etaB / (dx * dx)

#             # Cross terms with vy
#             vy1_coeff =  eta1 / (dx * dy)
#             vy2_coeff = -eta2 / (dx * dy)
#             vy3_coeff = -eta1 / (dx * dy)
#             vy4_coeff =  eta2 / (dx * dy)

#             # 3) Sum of off-diagonal contributions
#             x_mom = (
#                 vx1_coeff * vx[i,   j-1] +
#                 vx2_coeff * vx[i-1, j  ] +
#                 vx3_coeff * vx[i,   j  ] +
#                 vx4_coeff * vx[i+1, j  ] +
#                 vx5_coeff * vx[i,   j+1]
#                 +
#                 vy1_coeff * vy[i-1, j  ] +
#                 vy2_coeff * vy[i,   j  ] +
#                 vy3_coeff * vy[i-1, j+1] +
#                 vy4_coeff * vy[i,   j+1]
#             )
            
#             rax[i, j] = x_mom

#     # Need to simulate effect of identity operator
#     # and boundary operator on the input
#     rax[:, 0] = vx[:, 0]                        # Left boundary
#     rax[:, -2:] = vx[:, -2:]                    # Right boundary
#     rax[0, :-1] = vx[0, :-1] + BC*vx[1, :-1]    # Top boundary
#     rax[-1, :-1] = vx[-1, :-1] + BC*vx[-1, :-1] # Bottom boundary

# @nb.njit(cache=True, parallel=True)
# def fixed_p_vy_operator(nx1, ny1,
#                         dx, dy,
#                         etap, etab,
#                         vx, vy, rax, BC):
#     """
#     Compute the y-momentum residual for each interior cell.
#     Returns res_y, a 2D array of the same shape as vy.
#     """
#     # Loop over interior
#     for i in nb.prange(1, ny1 - 2):
#         for j in nb.prange(1, nx1 - 1):

#             # 1) Local viscosities
#             etaA = etap[i,   j]      # top cell's viscosity
#             etaB = etap[i+1, j]      # bottom cell's viscosity
#             eta1 = etab[i,   j-1]    
#             eta2 = etab[i,   j]

#             # 2) Coefficients for y-momentum operator
#             vy1_coeff = eta1 / (dx * dx)
#             vy2_coeff = 2.0 * etaA / (dy * dy)
#             vy3_coeff = (-2.0 * etaA / (dy * dy)
#                          -2.0 * etaB / (dy * dy)
#                          -eta1 / (dx * dx)
#                          -eta2 / (dx * dx))
#             vy4_coeff = 2.0 * etaB / (dy * dy)
#             vy5_coeff = eta2 / (dx * dx)

#             # Cross terms with vx
#             vx1_coeff =  eta1 / (dx * dy)
#             vx2_coeff = -eta1 / (dx * dy)
#             vx3_coeff = -eta2 / (dx * dy)
#             vx4_coeff =  eta2 / (dx * dy)

#             # 3) Sum of off-diagonal contributions
#             y_mom = (
#                 vy1_coeff * vy[i,   j-1] +
#                 vy2_coeff * vy[i-1, j  ] +
#                 vy3_coeff * vy[i,   j  ] +
#                 vy4_coeff * vy[i+1, j  ] +
#                 vy5_coeff * vy[i,   j+1]
#                 +
#                 vx1_coeff * vx[i,   j-1] +
#                 vx2_coeff * vx[i+1, j-1] +
#                 vx3_coeff * vx[i,   j  ] +
#                 vx4_coeff * vx[i+1, j  ]
#             )

#             rax[i, j] = y_mom

#     # Need to simulate effect of identity operator
#     # and boundary operator on the input
#     rax[0, :] = vy[0, :]                        # Top boundary
#     rax[-2:, :] = vy[-2:, :]                    # Bottom boundary
#     rax[:-1, 0] = vy[:-1, 0] + BC*vy[:-1, 1]    # Left boundary
#     rax[:-1, -1] = vy[:-1, -1] + BC*vy[:-1, -1] # Right boundary