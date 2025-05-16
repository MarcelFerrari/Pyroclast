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

@timer.time_function("Model Solve", "Restriction")
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

    for t in nb.prange(nt):
        start = t*nyh//nt
        end = (t+1)*nyh//nt if t < nt-1 else nyh
        for i in range(start, end):
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

@timer.time_function("Model Solve", "Prolongation")
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
