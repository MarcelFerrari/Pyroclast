"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: mg_routines.py
Description: This file implements restriction and prolongation operators for
             multigrid methods.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from Pyroclast.profiling import timer
import numba as nb
import numpy as np

@nb.njit(cache=True, inline='always')
def clip(x, xmin, xmax):
    """
    Clip a value x to the range [a, b].
    """
    if x < xmin:
        return xmin
    elif x > xmax:
        return xmax
    else:
        return x

@nb.njit(cache=True, parallel=True)
def restrict(xh, yh, uh, xH, yH, nt = nb.get_num_threads()):
    """
    Restriction operator for multigrid method.
    xh: fine grid x-coordinates
    yh: fine grid y-coordinates
    uh: fine grid quantity
    xH: coarse grid x-coordinates
    yH: coarse grid y-coordinates
    uH: coarse grid quantity
    """
    nxh = len(xh)
    nyh = len(yh)
    nxH = len(xH)
    nyH = len(yH)
    dxH = xH[1] - xH[0]
    dyH = yH[1] - yH[0]
    xH0 = xH[0]
    yH0 = yH[0]

    # Allocate memory
    uH = np.zeros((nt, nyH, nxH))
    uHw = np.zeros((nt, nyH, nxH))

    # MEGA IMPORTANT:
    # The loop on the fine grid should be from 0 to nyh - 1 and from 0 to nxh - 1
    # This is because otherwise we interpolate the values at the edges of the fine grid
    # which ALWAYS correspond to ghost points in the coarse grid!!

    for t in nb.prange(nt):
        start = t*nyh//nt
        end = (t+1)*nyh//nt if t < nt-1 else nyh - 1
        for i in range(start, end):
            for j in range(nxh - 1):
                # Extract coordinates of source point
                yhi = yh[i]
                xhj = xh[j]
                
                # Find 4 nearest neighbors in coarse grid
                iH = clip(int((yhi - yH0)/dyH), 0, nyH-2)
                jH = clip(int((xhj - xH0)/dxH), 0, nxH-2)

                # Compute interpolation weights
                ry = (yhi - yH[iH])/dyH
                rx = (xhj - xH[jH])/dxH

                # Interpolate quantities
                uH[t, iH, jH] += (1-rx)*(1-ry)*uh[i, j]
                uH[t, iH+1, jH] += rx*(1-ry)*uh[i, j]
                uH[t, iH, jH+1] += (1-rx)*ry*uh[i, j]
                uH[t, iH+1, jH+1] += rx*ry*uh[i, j]

                # Store weights
                uHw[t, iH, jH] += (1-rx)*(1-ry)
                uHw[t, iH+1, jH] += rx*(1-ry)
                uHw[t, iH, jH+1] += (1-rx)*ry
                uHw[t, iH+1, jH+1] += rx*ry
    
    # Reduce uH and uHw
    uH = np.sum(uH, axis=0)
    uHw = np.sum(uHw, axis=0)

    # Normalize weights
    uH /= uHw

    return np.nan_to_num(uH)

@nb.njit(cache=True, parallel=True)
def prolong(xh, yh, xH, yH, uH):
    """
    Prolongation operator for multigrid method.
    xh: fine grid x-coordinates
    yh: fine grid y-coordinates
    uh: tuple of fine grid quantities
    xH: coarse grid x-coordinates
    yH: coarse grid y-coordinates
    uH: tuple of coarse grid quantities
    """
    nxh = len(xh)
    nyh = len(yh)
    
    nxH = len(xH)
    nyH = len(yH)
    dxH = xH[1] - xH[0]
    dyH = yH[1] - yH[0]
    xH0 = xH[0]
    yH0 = yH[0]

    # Allocate memory
    uh = np.zeros((nyh, nxh))

    for i in nb.prange(nyh):
        for j in nb.prange(nxh):
            # Extract coordinates of source point
            yhi = yh[i]
            xhj = xh[j]

            # Find 4 nearest neighbors in fine grid
            iH = clip(int((yhi - yH0)/dyH), 0, nyH-2)
            jH = clip(int((xhj - xH0)/dxH), 0, nxH-2)

            # Compute interpolation weights
            ry = (yhi - yH[iH])/dyH
            rx = (xhj - xH[jH])/dxH

            # Interpolate quantities
            uh[i, j] = (1-rx)*(1-ry)*uH[iH, jH] + \
                       rx*(1-ry)*uH[iH, jH+1] + \
                       (1-rx)*ry*uH[iH+1, jH] + \
                       rx*ry*uH[iH+1, jH+1]
    
    return np.nan_to_num(uh)

@nb.njit(cache=True, parallel=True)
def uzawa_velocity_rhs(nx1, ny1,
                       dx, dy,
                       vx_rhs, vy_rhs, p,
                       vx_rhs_uzawa, vy_rhs_uzawa):
    
    # vx rhs
    for i in nb.prange(1, ny1 - 1):
        for j in nb.prange(1, nx1 - 2):
            # Uzawa rhs = vx_rhs + dP/dx
            vx_rhs_uzawa[i, j] = vx_rhs[i, j] + (p[i, j+1] - p[i, j]) / dx

    # vy rhs
    for i in nb.prange(1, ny1 - 2):
        for j in nb.prange(1, nx1 - 1):
            # Uzawa rhs = vy_rhs + dP/dy
            vy_rhs_uzawa[i, j] = vy_rhs[i, j] + (p[i+1, j] - p[i, j]) / dy
    
    return vx_rhs_uzawa, vy_rhs_uzawa

@nb.njit(cache=True, parallel=True)
def uzawa_vx_operator(nx1, ny1,
                dx, dy,
                etap, etab,
                vx, vy, rax):
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
    return rax

@nb.njit(cache=True, parallel=True)
def uzawa_vx_residual(nx1, ny1,
                dx, dy,
                etap, etab,
                vx, vy, res_vx, rhs):
    """
    Compute the x-momentum residual for each interior cell.
    Returns res_x, a 2D array of the same shape as vx.
    """
    
    # Store operator result in res_vx
    res_vx = uzawa_vx_operator(nx1, ny1, dx, dy, etap, etab, vx, vy, res_vx)

    # Compute residual
    for i in nb.prange(1, ny1 - 1):
        for j in nb.prange(1, nx1 - 2):
            res_vx[i, j] = rhs[i, j] - res_vx[i, j]
    
    return res_vx

@nb.njit(cache=True, parallel=True)
def uzawa_vy_operator(nx1, ny1,
                      dx, dy,
                      etap, etab,
                      vx, vy, rax):
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
    return rax

@nb.njit(cache=True, parallel=True)
def uzawa_vy_residual(nx1, ny1,
                      dx, dy,
                      etap, etab,
                      vx, vy, res_vy, rhs):
    """
    Compute the y-momentum residual for each interior cell.
    Returns res_y, a 2D array of the same shape as vy.
    """
    # Store operator result in res_vy
    res_vy = uzawa_vy_operator(nx1, ny1,
                               dx, dy,
                               etap, etab,
                               vx, vy, res_vy)

    # Compute residual
    for i in nb.prange(1, ny1 - 2):
        for j in nb.prange(1, nx1 - 1):
            res_vy[i, j] = rhs[i, j] - res_vy[i, j]
    
    return res_vy