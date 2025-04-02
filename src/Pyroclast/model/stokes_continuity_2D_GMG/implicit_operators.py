"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: implicit_operators.py
Description: This file implements the implicit operators for the Stokes flow
             and continuity equations in 2D.
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

@nb.njit(cache=True, parallel=True)
def fixed_p_vx_operator(nx1, ny1,
                        dx, dy,
                        etap, etab,
                        vx, vy, rax, BC):
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
            )
            
            rax[i, j] = x_mom

    # Need to simulate effect of identity operator
    # and boundary operator on the input
    rax[:, 0] = vx[:, 0]                        # Left boundary
    rax[:, -2:] = vx[:, -2:]                    # Right boundary
    rax[0, :-1] = vx[0, :-1] + BC*vx[1, :-1]    # Top boundary
    rax[-1, :-1] = vx[-1, :-1] + BC*vx[-1, :-1] # Bottom boundary

@nb.njit(cache=True, parallel=True)
def fixed_p_vy_operator(nx1, ny1,
                        dx, dy,
                        etap, etab,
                        vx, vy, rax, BC):
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
            )

            rax[i, j] = y_mom

    # Need to simulate effect of identity operator
    # and boundary operator on the input
    rax[0, :] = vy[0, :]                        # Top boundary
    rax[-2:, :] = vy[-2:, :]                    # Bottom boundary
    rax[:-1, 0] = vy[:-1, 0] + BC*vy[:-1, 1]    # Left boundary
    rax[:-1, -1] = vy[:-1, -1] + BC*vy[:-1, -1] # Right boundary

@nb.njit(cache=True, parallel=True)
def momentum_operator(x, nx1, ny1, dx, dy, etap, etab, BC, Ax):    
    for i in nb.prange(ny1):
        for j in range(nx1):
            # 2) x-momentum equation (vx)
            if j == 0 or j >= nx1-2: # Last two nodes in x-direction
                Ax[i, j, 0] = x[i, j, 0]

            elif i == 0 or i == ny1 - 1: # First and last nodes in y-direction
                if i == 0: # Top boundary
                    Ax[i, j, 0] = x[i, j, 0] + BC*x[i+1, j, 0]
                else: # Bottom boundary
                    Ax[i, j, 0] = x[i, j, 0] + BC*x[i-1, j, 0]
            else:
                # Extract viscosity values
                etaA = etap[i, j]
                etaB = etap[i, j+1]
                eta1 = etab[i-1, j]
                eta2 = etab[i, j]

                # vx coefficients
                vx1_coeff = 2*etaA/dx**2
                vx2_coeff = eta1/dy**2
                vx3_coeff = -eta1/dy**2 - eta2/dy**2 - 2*etaA/dx**2 - 2*etaB/dx**2
                vx4_coeff = eta2/dy**2
                vx5_coeff = 2*etaB/dx**2

                # vy coefficients
                vy1_coeff = eta1/(dx*dy)
                vy2_coeff = -eta2/(dx*dy)
                vy3_coeff = -eta1/(dx*dy)
                vy4_coeff = eta2/(dx*dy)

                Ax[i, j, 0] = (vx1_coeff*x[i, j-1, 0] +
                               vx2_coeff*x[i-1, j, 0] +
                               vx3_coeff*x[i, j, 0] +
                               vx4_coeff*x[i+1, j, 0] +
                               vx5_coeff*x[i, j+1, 0] +
                               vy1_coeff*x[i-1, j, 1] +
                               vy2_coeff*x[i, j, 1] +
                               vy3_coeff*x[i-1, j+1, 1] +
                               vy4_coeff*x[i, j+1, 1])
                
            # 3) y-momentum equation (vy)
            if i == 0 or i >= ny1 - 2: # Last two nodes in y-direction
                Ax[i, j, 1] = x[i, j, 1]
            elif j == 0 or j == nx1 - 1:
                if j == 0:
                    Ax[i, j, 1] = x[i, j, 1] + BC*x[i, j+1, 1]
                else:
                    Ax[i, j, 1] = x[i, j, 1] + BC*x[i, j-1, 1]
            else:
                # Extract viscosity values
                etaA = etap[i, j]
                etaB = etap[i+1, j]
                eta1 = etab[i, j-1]
                eta2 = etab[i, j]

                # vy coefficients
                vy1_coeff = eta1/dx**2
                vy2_coeff = 2*etaA/dy**2
                vy3_coeff = -2*etaA/dy**2 - 2*etaB/dy**2 - eta1/dx**2 - eta2/dx**2
                vy4_coeff = 2*etaB/dy**2
                vy5_coeff = eta2/dx**2

                #vx coefficients
                vx1_coeff = eta1/(dx*dy)
                vx2_coeff = -eta1/(dx*dy)
                vx3_coeff = -eta2/(dx*dy)
                vx4_coeff = eta2/(dx*dy)

                Ax[i, j, 1] = (vy1_coeff*x[i, j-1, 1] +
                               vy2_coeff*x[i-1, j, 1] +
                               vy3_coeff*x[i, j, 1] +
                               vy4_coeff*x[i+1, j, 1] +
                               vy5_coeff*x[i, j+1, 1] +
                               vx1_coeff*x[i, j-1, 0] +
                               vx2_coeff*x[i+1, j-1, 0] +
                               vx3_coeff*x[i, j, 0] +
                               vx4_coeff*x[i+1, j, 0])
                
# Assemble rhs of momentum equations assuming fixed pressure
@nb.njit(cache = True)
def assemble_momentum_rhs(nx1, ny1, dx, dy, rho, gy, p):
    b = np.zeros((ny1, nx1, 2), dtype=np.float64)
    
    for i in range(ny1):
        for j in range(nx1):
            # 2) x-momentum equation (vx)
            if j == 0 or j >= nx1-2: # Last two nodes in x-direction
                b[i, j, 0] = 0.0
            elif i == 0 or i == ny1 - 1: # First and last nodes in y-direction
                b[i, j, 0] = 0.0
            else:        
                # RHS
                b[i, j, 0] = (p[i, j+1] - p[i, j])/dx

            # 3) y-momentum equation (vy)
            if i == 0 or i >= ny1 - 2: # Last two nodes in y-direction
                b[i, j, 1] = 0.0
            elif j == 0 or j == nx1 - 1:
                b[i, j, 1] = 0.0
            else:
                # RHS
                b[i, j, 1] = -gy*rho[i, j] + (p[i+1, j] - p[i, j])/dy
    return b