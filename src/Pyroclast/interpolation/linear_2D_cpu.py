"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: 2D_linear_cpu.py
Description: This file implements 2D linear interpolation functions for CPU.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import warnings
import numba as nb
import numpy as np
from Pyroclast.profiling import timer

from Pyroclast.interpolation.utils import bisect_idx, compute_idx

@nb.njit(parallel=True, cache=True)
def reduce_marker_values(x, y, xm, ym, xidx, yidx, vals, grid_values, grid_weights, n_threads):
    """
    Loops over each marker and computes the weighted sum of quantities for the surrounding grid nodes.

    Parameters:
    - x, y, z: 1D arrays defining the coordinates of the regular grid in 3D
    - xm, ym: 1D arrays of shape (n_markers,) containing marker coordinates 
    - xidx, yidx: 1D arrays of shape (n_markers,) containing the index of the reference node for each marker
    - vals: tuple of 1D arrays of shape (n_markers,) containing marker values to interpolate
    - grid_values: tuple of 3D arrays of shape (n_threads, len(x), len(y)) with interpolated values at grid nodes
    - grid_weights: 3D zeros array (n_threads, len(x), len(y)).

    Returns:
    - grid_values: tuple of 3D arrays (n_threads, len(x), len(y)) with interpolated values at grid nodes
    - grid_weights: 3D array (n_threads, len(x), len(y)) with accumulated weights at grid nodes
    """

    # Get dimensions of tensors
    n_markers = len(xm)
    dx, dy = x[1] - x[0], y[1] - y[0]

    # Loop over each marker in parallel
    for t in nb.prange(n_threads):
        # Get thread-specific marker indices
        start = t * n_markers // n_threads
        end = (t + 1) * n_markers // n_threads if t != n_threads - 1 \
                                               else n_markers
        
        for m in range(start, end):
            # Read marker coordinates and reference node indices
            mx, my = xm[m], ym[m]
            mj, mi = xidx[m], yidx[m]
            
            # Loop over each surrounding grid node within the voxel
            for offset_x in range(2):
                for offset_y in range(2):
                    # Determine grid node indices
                    gx, gy = mj + offset_x, mi + offset_y

                    # Calculate distances from marker to current grid node
                    rx = np.abs(mx - x[gx])
                    ry = np.abs(my - y[gy])

                    # Calculate weight based on distance to this nodal point
                    w = (1 - rx / dx) * (1 - ry / dy)

                    # Update weighted sums for quantities and weights at this grid point
                    for q in range(len(vals)):
                        grid_values[q][t, gy, gx] += w * vals[q][m]
                    
                    grid_weights[t, gy, gx] += w
    
    return grid_values, grid_weights

def interpolate_markers2grid(x, y, xm, ym, vals, indexing="bisect", return_weights=False, ghost_nodes=True, real=np.float64):
    """
    Interpolates marker values to the grid nodes using distance-weighted linear interpolation.
    
    Parameters:
    - x, y: 1D arrays defining the coordinates of the regular grid in 2D
    - xm, ym: 1D arrays of shape (n_markers,) containing marker coordinates
    - vals: tuple of 1D arrays of shape (n_markers,) containing marker values to interpolate
    - indexing: str, optional, default: "bisect". Indexing mode for grid nodes.
                "equidistant": grid nodes are equidistantly spaced
                "bisect": grid nodes are non-equidistantly spaced and indices are computed by bisection
    - return_weights: bool, optional, default: False. If True, returns the accumulated weights at each grid node
    - real: np.dtype, optional, default: np.float64. Real type for the arrays

    Returns:
    If return_weights is True:
    - grid_values: tuple of 2D arrays of shape (len(x), len(y)) with normalized interpolated values at grid nodes
    If return_weights is False:
    - grid_values: tuple of 2D arrays of shape (len(x), len(y)) with non-normalized interpolated values at grid nodes
    - grid_weights: 3D array (len(x), len(y), len(z)) with accumulated weights at grid nodes
    """

    # Check input classes
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(xm, np.ndarray)
    assert isinstance(ym, np.ndarray)
    assert isinstance(vals, tuple)
    
    # Check input data types
    assert x.dtype == real
    assert y.dtype == real
    assert xm.dtype == real
    assert ym.dtype == real

    # Check input shapes
    assert x.ndim == 1
    assert y.ndim == 1
    assert xm.ndim == 1
    assert ym.ndim == 1
    assert len(vals) > 0

    # Dimensions of the grid
    nx, ny, q = len(x), len(y), len(vals)
    
    # Get number of threads
    n_threads = nb.get_num_threads()

    # Initialize grid_values and grid_weights
    # Allocate leading dimension for number of threads
    grid_values = tuple(np.zeros((n_threads, ny, nx), dtype=real) for _ in range(q))
    grid_weights = np.zeros((n_threads, ny, nx), dtype=real)

    # 1) Compute grid indices for each marker
    # Important to pass nx and ny to the indexing functions
    # in order to handle ghost nodes correctly
    if indexing == "equidistant":
        xidx = compute_idx(x, xm)
        yidx = compute_idx(y, ym)
    elif indexing == "bisect":
        xidx = bisect_idx(x, xm, nx)
        yidx = bisect_idx(y, ym, nx)
    else:
        raise ValueError("Invalid indexing mode. Choose 'equidistant' or 'bisect'.")

    # 2) Loop over markers and accumulate weighted values and weights
    grid_values, grid_weights = reduce_marker_values(x, y, xm, ym, xidx, yidx, vals, grid_values, grid_weights, n_threads)

    # Reduce partial results from each thread
    grid_values = tuple(np.sum(v, axis=0) for v in grid_values)
    grid_weights = np.sum(grid_weights, axis=0)

    if return_weights: # We are done
        return grid_values, grid_weights
    else: # Normalize grid values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Ignore division by zero!
            # This is correct and represents values outside the grid!
            rax = tuple(v / grid_weights for v in grid_values)
        return rax if len(rax) > 1 else rax[0] # Return a single value if only one quantity is interpolated

@nb.njit(parallel=True, cache=True)
def reduce_grid_values(x, y, xm, ym, xidx, yidx, grid_values, marker_values):
    """
    Reduces the grid values to the markers using distance-weighted linear interpolation.

    Parameters:
    - x, y: 1D arrays defining the coordinates of the regular grid in 2D
    - xm, ym: 1D arrays of shape (n_markers,) containing marker coordinates
    - xidx, yidx: 1D arrays of shape (n_markers,) containing the index of the reference node for each marker
    - grid_values: tuple of 2D arrays of shape (len(x), len(y)) containing grid values to interpolate
    - marker_values: tuple of 1D arrays of shape (n_markers,) with interpolated values at markers
    - marker_weights: 1D array of shape (n_markers,) containing accumulated weights at each marker.
    - num_threads: int. Number of threads to use in the parallel loop.

    Returns:
    - marker_values: tuple of 1D arrays of shape (n_markers,) with interpolated values at markers
    - marker_weights: 1D array of shape (n_markers,) containing accumulated weights at each marker.
    """
    # Get dimensions of tensors
    n_markers = len(xm)
    dx, dy = x[1] - x[0], y[1] - y[0]
    q = len(grid_values)

    # Loop over each marker in parallel
    for m in nb.prange(n_markers):
        # Read marker coordinates and reference node indices
        mx, my = xm[m], ym[m]
        mj, mi = xidx[m], yidx[m]

        for x_offset in range(2):
            for y_offset in range(2):

                    # Determine grid node indices
                    gx, gy = mj + x_offset, mi + y_offset

                    # Calculate distances from marker to current grid node
                    rx = np.abs(mx - x[gx])
                    ry = np.abs(my - y[gy])

                    # Calculate weight based on distance to this nodal point
                    w = (1 - rx / dx) * (1 - ry / dy)

                    # Update weighted sums for quantities and weights at this grid point
                    for q in range(len(grid_values)):
                        marker_values[q][m] += w * grid_values[q][gy, gx]
    return marker_values


@nb.njit(parallel=True, cache=True)
def apply_x_continuity_correction(x, y, xm, ym, xidx, yidx, vx, vxm, d2dx2):
    """
    Applies continuity correction in x to the interpolated values at the markers.

    Parameters:
    - x, y: 1D arrays defining the coordinates of the regular grid in 2D
    - xm, ym: 1D arrays of shape (n_markers,) containing marker coordinates
    - xidx, yidx: 1D arrays of shape (n_markers,) containing the index of the reference node for each marker
    - vx: 2D array of shape (len(x), len(y)) describing velocity in x direction
    - vxm: 1D array of shape (n_markers,) with interpolated velocities values at markers
    - d2dx2: 2D array of shape (len(x), len(y)) with second derivative of velocity in x direction

    Returns:
    - marker_values: tuple of 1D arrays of shape (n_markers,) with corrected values at markers
    """
     # Get dimensions of tensors
    n_markers = len(xm)
    nx, ny = len(x), len(y)
    dx, dy = x[1] - x[0], y[1] - y[0]

    # Precompute second derivative of velocity in x direction
    for i in nb.prange(ny):
        for j in nb.prange(1, nx-1):
            d2dx2[i, j] = vx[i, j-1] - 2*vx[i, j] + vx[i, j+1]

    # Loop over each marker in parallel
    for m in nb.prange(n_markers):
        mi, mj = yidx[m], xidx[m]
        mx, my = xm[m], ym[m]

        # Magic correction term
        if (mj > 1 and mx <= (dx/2 + x[mj])) or (mj < nx-2 and mx > (dx/2 + x[mj])):
            # Calculate distances from marker to current grid node
            rx = np.abs(mx - x[mj])
            ry = np.abs(my - y[mi])

            if xm[m] > (dx/2 + x[mj]): # right side of cell
                vxm[m] += (1 - ry/dy) * (rx/dx - 0.5)**2 * 1/2 * d2dx2[mi, mj+1] \
                          + (ry/dy) * (rx/dx - 0.5)**2 * 1/2 * d2dx2[mi+1, mj+1]
            else: # left side of cell
                vxm[m] += (1 - ry/dy) * (rx/dx - 0.5)**2 * 1/2 * d2dx2[mi, mj] \
                          + (ry/dy) * (rx/dx - 0.5)**2 * 1/2 * d2dx2[mi+1, mj]
    
    return (vxm,)
    

@nb.njit(parallel=True, cache=True)
def apply_y_continuity_correction(x, y, xm, ym, xidx, yidx, vy, vym, d2dy2):
    """
    Applies continuity correction in y to the interpolated values at the markers.

    Parameters:
    - x, y: 1D arrays defining the coordinates of the regular grid in 2D
    - xm, ym: 1D arrays of shape (n_markers,) containing marker coordinates
    - xidx, yidx: 1D arrays of shape (n_markers,) containing the index of the reference node for each marker
    - vy: 2D array of shape (len(x), len(y)) describing velocity in y direction
    - vym: 1D array of shape (n_markers,) with interpolated velocities values at markers
    - d2dy2: 2D array of shape (len(x), len(y)) with second derivative of velocity in y direction

    Returns:
    - marker_values: tuple of 1D arrays of shape (n_markers,) with corrected values at markers
    """
     # Get dimensions of tensors
    n_markers = len(xm)
    nx, ny = len(x), len(y)
    dx, dy = x[1] - x[0], y[1] - y[0]

    # Precompute second derivative of velocity in x direction
    for i in nb.prange(1, ny-1):
        for j in nb.prange(nx):
            d2dy2[i, j] = vy[i+1, j] - 2*vy[i, j] + vy[i-1, j]

    # Loop over each marker in parallel
    for m in nb.prange(n_markers):
        mi, mj = yidx[m], xidx[m]
        mx, my = xm[m], ym[m]

        if (mi > 1 and ym[m] <= (dy/2 + y[mi])) or (mi < ny-2 and ym[m] > (dy/2 + y[mi])):
            #% Add correction term
            rx = np.abs(mx - x[mj])
            ry = np.abs(my - y[mi])

            # Magic correction term
            if ym[m] > (dy/2 + y[mi]): # top side of cell
                vym[m] += (1 - rx/dx) * (ry/dy - 0.5)**2 * 1/2 * d2dy2[mi+1, mj] \
                          + (rx/dx) * (ry/dy - 0.5)**2 * 1/2 * d2dy2[mi+1, mj+1]
            else: # bottom side of cell
                vym[m] += (1 - rx/dx) * (ry/dy - 0.5)**2 * 1/2 * d2dy2[mi, mj] \
                          + (rx/dx) * (ry/dy - 0.5)**2 * 1/2 * d2dy2[mi, mj+1]
    
    return (vym,)

def interpolate_grid2markers(x, y, xm, ym, grid_values, indexing="bisect", cont_corr = None, real=np.float64):
    """
    Interpolates grid values to the markers using distance-weighted trilinear interpolation.
    
    Parameters:
    - x, y, z: 1D arrays defining the coordinates of the regular grid in 2D
    - grid_values: tuple of 2D arrays of shape (len(x), len(y)) containing grid values to interpolate
    - indexing: str, optional, default: "bisect". Indexing mode for grid nodes.
                "equidistant": grid nodes are equidistantly spaced
                "bisect": grid nodes are non-equidistantly spaced and indices are computed by bisection
    - cont_corr: apply continuity correction for velocity based interpolation (None, "x" or "y")
    - return_weights: bool, optional, default: False. If True, returns the accumulated weights at each marker
                as well as the non-normalized interpolated values.
    - real: np.dtype, optional, default: np.float64. Real type for the arrays
    
    Returns:
    If return_weights is True:
    - marker_values: tuple of 1D arrays of shape (n_markers,) with normalized interpolated values at markers
    If return_weights is False:
    - marker_values: tuple of 1D arrays of shape (n_markers,) with non-normalized interpolated values at markers
    - marker_weights: 1D array of shape (n_markers,) containing accumulated weights at each marker
    """
    
    # Assert that input arrays are CuPy arrays and have dtype = real
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(xm, np.ndarray)
    assert isinstance(ym, np.ndarray)
    assert isinstance(grid_values, tuple)
    assert cont_corr in [None, "x", "y"]

    # Grid dimensions and grid spacing
    nx, ny = len(x), len(y)

    # Initialize marker values and weights
    n_markers = len(xm)
    q = len(grid_values)

    marker_values = tuple(np.zeros((n_markers,), dtype=real) for _ in range(q))

    # 1) Compute grid indices for each marker
    if indexing == "equidistant":
        xidx = compute_idx(x, xm)
        yidx = compute_idx(y, ym)
    elif indexing == "bisect":
        xidx = bisect_idx(x, xm, nx)
        yidx = bisect_idx(y, ym, ny)
    else:
        raise ValueError("Invalid indexing mode. Choose 'equidistant' or 'bisect'.")
    
    # 2) Loop over markers and accumulate weighted values and weights
    marker_values = reduce_grid_values(x, y,
                                       xm, ym,
                                       xidx, yidx,
                                       grid_values,
                                       marker_values)
    
    # If necessary, apply continuity correction
    if cont_corr == "x": # Apply continuity correction in x
        d2dx2 = np.zeros((ny, nx), dtype=real)
        marker_values = apply_x_continuity_correction(x, y,
                                                      xm, ym,
                                                      xidx, yidx,
                                                      grid_values[0],
                                                      marker_values[0],
                                                      d2dx2)    
    elif cont_corr == "y": # Apply continuity correction in y
        d2dy2 = np.zeros((ny, nx), dtype=real)
        marker_values = apply_y_continuity_correction(x, y,
                                                      xm, ym,
                                                      xidx, yidx,
                                                      grid_values[0],
                                                      marker_values[0],
                                                      d2dy2)

    # Unpack the marker_values tuple if only one quantity is interpolated
    if len(marker_values) > 1:
        return marker_values
    else:
        return marker_values[0]