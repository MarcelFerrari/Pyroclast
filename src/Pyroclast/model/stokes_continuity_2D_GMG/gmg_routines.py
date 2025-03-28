"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: gmg_routines.py
Description: This file implements restriction and prolongation operators for
             multigrid methods.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numba as nb
import numpy as np

@nb.njit(cache=True)
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

@nb.njit(cache=True)
def restrict(xh, yh, uh, xH, yH):
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
    uH = np.zeros((nyH, nxH))
    uHw = np.zeros((nyH, nxH))

    for i in range(nyh):
        for j in range(nxh):
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
            uH[iH, jH] += (1-rx)*(1-ry)*uh[i, j]
            uH[iH+1, jH] += rx*(1-ry)*uh[i, j]
            uH[iH, jH+1] += (1-rx)*ry*uh[i, j]
            uH[iH+1, jH+1] += rx*ry*uh[i, j]

            # Store weights
            uHw[iH, jH] += (1-rx)*(1-ry)
            uHw[iH+1, jH] += rx*(1-ry)
            uHw[iH, jH+1] += (1-rx)*ry
            uHw[iH+1, jH+1] += rx*ry

    # Normalize weights
    uH /= uHw

    return np.nan_to_num(uH)

@nb.njit(cache=True)
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

    for i in range(nyh):
        for j in range(nxh):
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
