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


import os

import numba as nb
import numpy as np

from Pyroclast.utils import clip

use_fast_math_cpu = os.environ.get("PYROCLAST_FASTMATH_CPU", default=False)



@nb.njit(cache=True, inline="always", fastmath=use_fast_math_cpu)
def restrict_loop_body(xh: np.ndarray, yh: np.ndarray, uh: np.ndarray,
                       nxH: np.ndarray, nyH: int,
                       xH: np.ndarray, yH: np.ndarray, uH: np.ndarray, uHw: np.ndarray,
                       xH0: float, yH0: float, dxH: float, dyH: float , i: int, j: int):
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



@nb.njit(cache=True, fastmath=use_fast_math_cpu)
def restrict_2D(nxh: int, nyh: int, xh: np.ndarray, yh: np.ndarray, uh: np.ndarray,
                nxH: int, nyH: int, xH: np.ndarray, yH: np.ndarray, uH: np.ndarray, uHw: np.ndarray):
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
            restrict_loop_body(xh=xh, yh=yh, uh=uh,
                               nxH=nxH, nyH=nyH, xH=xH, yH=yH, uH=uH, uHw=uHw,
                               xH0=xH0, yH0=yH0, dxH=dxH, dyH=dyH, i=i, j=j)

    uH[:, :] /= uHw[:, :]

    return uH


@nb.njit(cache=True, parallel=True, fastmath=use_fast_math_cpu)
def restrict_2D_parallel_(nxh: int, nyh: int, xh: np.ndarray, yh: np.ndarray, uh: np.ndarray,
                         nxH: int, nyH: int, xH: np.ndarray, yH: np.ndarray, uH: np.ndarray, uHw: np.ndarray, threads: int):
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

    # Problem is too small, default back to
    if threads > ny - 1:
        return restrict_2D(nxh=nxh, nyh=nyh, xh=xh, yh=yh, uh=uh,
                           nxH=nxH, nyH=nyH, xH=xH, yH=yH, uH=uH, uHw=uHw)

    # MEGA IMPORTANT:
    # The loop on the fine grid should be from 0 to nyh - 1 and from 0 to nxh - 1
    # This is because otherwise we interpolate the values at the edges of the fine grid
    # which ALWAYS correspond to ghost points in the coarse grid!!
    thread_partition = (nyh - 1) // threads
    for c in range(4):
        for t in nb.prange(threads):
            ts = t * thread_partition
            te = (t + 1) * thread_partition if t + 1 < threads else nyh - 1

            for i in range(ts, te):
                for j in range(nxh - 1):
                    yhi = yh[i]
                    xhj = xh[j]

                    # Find 4 nearest neighbors in coarse grid
                    iH = clip(int((yhi - yH0) / dyH), 0, nyH - 2)
                    jH = clip(int((xhj - xH0) / dxH), 0, nxH - 2)

                    # check the color of the outer most loop.
                    if 2 * (iH % 2) + (jH % 2) != c:
                        continue

                    restrict_loop_body(xh=xh, yh=yh, uh=uh,
                                       nxH=nxH, nyH=nyH, xH=xH, yH=yH, uH=uH, uHw=uHw,
                                       xH0=xH0, yH0=yH0, dxH=dxH, dyH=dyH, i=i, j=j)


    uH[:, :] /= uHw[:, :]

    return uH


def restrict_2D_parallel(nxh: np.ndarray, nyh: np.ndarray,
                         xh: np.ndarray, yh: np.ndarray, uh: np.ndarray,
                         nxH: np.ndarray, nyH: np.ndarray,
                         xH: np.ndarray, yH: np.ndarray, uH: np.ndarray, uHw: np.ndarray, threads: int):
    uH = restrict_2D_parallel_(nxh, nyh, xh, yh, uh, nxH, nyH, xH, yH, uH, uHw, threads)
    uH1 = np.copy(uH)
    uH2 = restrict_2D(nxh, nyh, xh, yh, uh, nxH, nyH, xH, yH, uH, uHw)
    print(np.sum(np.abs(uH1 - uH2)))


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
