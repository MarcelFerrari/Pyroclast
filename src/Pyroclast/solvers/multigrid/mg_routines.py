"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/solvers/multigrid/mg_routines.py  
Description: File contains routines to deal with grid levels - restrict and prolong.

Author: Marcel Ferrari, Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


import numba as nb

from Pyroclast.utils import clip


@nb.njit(cache=True)
def restrict_2D(nxh, nyh, xh, yh, uh, nxH, nyH, xH, yH, uH, uHw):
    """
    Restriction operator for multigrid method.
    xh: fine grid x-coordinates
    yh: fine grid y-coordinates
    uh: fine grid quantity
    xH: coarse grid x-coordinates
    yH: coarse grid y-coordinates
    uH: coarse grid quantity
    """
    dxH = xH[1] - xH[0]
    dyH = yH[1] - yH[0]
    xH0 = xH[0]
    yH0 = yH[0]

    uH[:, :] = 0.0
    uHw[:, :] = 0.0

    # MEGA IMPORTANT:
    # The loop on the fine grid should be from 0 to nyh - 1 and from 0 to nxh - 1
    # This is because otherwise we interpolate the values at the edges of the fine grid
    # which ALWAYS correspond to ghost points in the coarse grid!!

    for i in range(nyh - 1):
        for j in range(nxh - 1):
            # Extract coordinates of source point
            yhi = yh[i]
            xhj = xh[j]

            # Find 4 nearest neighbors in coarse grid
            iH = clip(int((yhi - yH0) / dyH), 0, nyH - 2)
            jH = clip(int((xhj - xH0) / dxH), 0, nxH - 2)

            # Compute interpolation weights
            ry = (yhi - yH[iH]) / dyH
            rx = (xhj - xH[jH]) / dxH

            # Interpolate quantities
            uH[iH, jH] += (1 - rx) * (1 - ry) * uh[i, j]
            uH[iH + 1, jH] += (1 - rx) * ry * uh[i, j]
            uH[iH, jH + 1] += rx * (1 - ry) * uh[i, j]
            uH[iH + 1, jH + 1] += rx * ry * uh[i, j]

            # Store weights
            uHw[iH, jH] += (1 - rx) * (1 - ry)
            uHw[iH + 1, jH] += (1 - rx) * ry
            uHw[iH, jH + 1] += rx * (1 - ry)
            uHw[iH + 1, jH + 1] += rx * ry

    uH[:, :] /= uHw[:, :]

    return uH


@nb.njit(cache=True, parallel=True)
def prolong_2D(nxH, nyH, xH, yH, uH,
               nxh, nyh, xh, yh, uh):
    """
    Prolongation operator for multigrid method.
    xh: fine grid x-coordinates
    yh: fine grid y-coordinates
    uh: tuple of fine grid quantities
    xH: coarse grid x-coordinates
    yH: coarse grid y-coordinates
    uH: tuple of coarse grid quantities
    """

    dxH = xH[1] - xH[0]
    dyH = yH[1] - yH[0]
    xH0 = xH[0]
    yH0 = yH[0]

    for i in nb.prange(nyh):
        for j in nb.prange(nxh):
            # Extract coordinates of source point
            yhi = yh[i]
            xhj = xh[j]

            # Find 4 nearest neighbors in fine grid
            iH = clip(int((yhi - yH0) / dyH), 0, nyH - 2)
            jH = clip(int((xhj - xH0) / dxH), 0, nxH - 2)

            # Compute interpolation weights
            ry = (yhi - yH[iH]) / dyH
            rx = (xhj - xH[jH]) / dxH

            # Interpolate quantities
            uh[i, j] = (1 - rx) * (1 - ry) * uH[iH, jH] + \
                       rx * (1 - ry) * uH[iH, jH + 1] + \
                       (1 - rx) * ry * uH[iH + 1, jH] + \
                       rx * ry * uH[iH + 1, jH + 1]

    return uh
