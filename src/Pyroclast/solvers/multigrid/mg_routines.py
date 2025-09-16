
from Pyroclast.profiling import timer
from Pyroclast.utils import clip

import numba as nb
import numpy as np


@nb.njit(cache=True, parallel=True)
def restrict_2D(xh, yh, uh, xH, yH, nt = nb.get_num_threads()):
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
def prolong_2D(xh, yh, xH, yH, uH):
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