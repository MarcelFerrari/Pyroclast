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

from Pyroclast.utils import clip, inject_threads

use_fast_math_cpu = os.environ.get("PYROCLAST_FASTMATH_CPU", default=False)



@nb.njit(inline="always")
def find_lower_index(xf: np.ndarray, target: float) -> int:
    """
    Return smallest index i such that xf[i] >= target.
    (Equivalent to np.searchsorted(xf, target, side="left"))
    """
    left = 0
    right = xf.shape[0] - 1
    while left < right:
        mid = (left + right) // 2
        if xf[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


@nb.njit(cache=True, inline="always", fastmath=use_fast_math_cpu)
def restrict_loop_body(xh: np.ndarray, yh: np.ndarray, uh: np.ndarray,
                       nxH: int, nyH: int,
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


@inject_threads
@nb.njit(cache=True, parallel=True, fastmath=use_fast_math_cpu)
def restrict_2D_parallel_(nxh: int, nyh: int, xh: np.ndarray, yh: np.ndarray, uh: np.ndarray,
                          nxH: int, nyH: int, xH: np.ndarray, yH: np.ndarray, uH: np.ndarray, uHw: np.ndarray,
                          th: int):
    """
    Restriction operator for multigrid method.
    xh: fine grid x-coordinates
    yh: fine grid y-coordinates
    uh: fine grid quantity
    xH: coarse grid x-coordinates
    yH: coarse grid y-coordinates
    uH: coarse grid quantity
    """
    i_fine_bounds = np.zeros((nyH), dtype=np.int32)
    j_fine_bounds = np.zeros((nxH), dtype=np.int32)

    dxH = float(xH[1] - xH[0])
    dyH = float(yH[1] - yH[0])

    xH0 = float(xH[0])
    yH0 = float(yH[0])

    uH[:, :] = 0.0
    uHw[:, :] = 0.0

    # Problem is too small, default back to
    # if th > nyH - 1:
    #     return restrict_2D(nxh=nxh, nyh=nyh, xh=xh, yh=yh, uh=uh,
    #                        nxH=nxH, nyH=nyH, xH=xH, yH=yH, uH=uH, uHw=uHw)

    # MEGA IMPORTANT:
    # The loop on the fine grid should be from 0 to nyh - 1 and from 0 to nxh - 1
    # This is because otherwise we interpolate the values at the edges of the fine grid
    # which ALWAYS correspond to ghost points in the coarse grid!!
    thread_partition_y = (nyH - 1) // th
    thread_partition_x = (nxH - 1) // th

    # Compute the bisections once in y
    for t in nb.prange(th):
        tsc = t * thread_partition_y
        tec = (t + 1) * thread_partition_y if t + 1 < th else nyH

        for i in range(tsc, tec):
            i_fine_bounds[i] = find_lower_index(yh, yH[i])


    # Compute bisection once in x
    for t in nb.prange(th):
        tsc = t * thread_partition_x
        tec = (t + 1) * thread_partition_x if t + 1 < th else nxH

        for i in range(tsc, tec):
            j_fine_bounds[i] = find_lower_index(xh, xH[i])

    for c in range(4):
        for t in nb.prange(th):
            tsc = t * thread_partition_y
            tec = (t + 1) * thread_partition_y if t + 1 < th else nyH - 1

            # Per thread iteration on the coarse grid
            for I in range(tsc, tec):
                for J in range(nxH - 1):

                    # check the color of the coarse grid.
                    if 2 * (I % 2) + (J % 2) != c:
                        continue

                    for i in range(i_fine_bounds[I], i_fine_bounds[I + 1]):
                        for j in range(j_fine_bounds[J], j_fine_bounds[J + 1]):
                            restrict_loop_body(xh=xh, yh=yh, uh=uh,
                                               nxH=nxH, nyH=nyH, xH=xH, yH=yH, uH=uH, uHw=uHw,
                                               xH0=xH0, yH0=yH0, dxH=dxH, dyH=dyH, i=i, j=j)


    uH[:, :] /= uHw[:, :]

    return uH


def restrict_2D_parallel(nxh: int, nyh: int,
                         xh: np.ndarray, yh: np.ndarray, uh: np.ndarray,
                         nxH: int, nyH: int,
                         xH: np.ndarray, yH: np.ndarray, uH: np.ndarray, uHw: np.ndarray):

    uH = restrict_2D_parallel_(nxh=nxh, nyh=nyh, xh=xh, yh=yh, uh=uh,
                               nxH=nxH, nyH=nyH, xH=xH, yH=yH, uH=uH, uHw=uHw)
    return uH
    # uH1 = np.copy(uH)
    # uH2 = restrict_2D(nxh, nyh, xh, yh, uh, nxH, nyH, xH, yH, uH, uHw)
    # print(np.sum(np.abs(uH1 - uH2)))


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
