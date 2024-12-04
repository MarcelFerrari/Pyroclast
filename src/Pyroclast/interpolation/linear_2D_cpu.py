"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: 2D_linear_cpu.py
Description: This file implements 2D linear interpolation functions for CPU.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

See LICENSE file in the project root for full license information.
"""

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

    # Assert that grid_weights is not zero
    # assert np.all(grid_weights > 0), "Some grid nodes have zero weight. Check marker positions."

    if return_weights: # We are done
        return grid_values, grid_weights
    else: # Normalize grid values
        rax = tuple(v / grid_weights for v in grid_values)
        return rax if len(rax) > 1 else rax[0] # Return a single value if only one quantity is interpolated

@nb.njit(parallel=True, cache=True)
def reduce_grid_values(x, y, xm, ym, xidx, yidx, grid_values, marker_values, marker_weights, num_threads):
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
    for t in nb.prange(num_threads):
        # Get thread-specific marker indices
        start = t * n_markers // num_threads
        end = (t + 1) * n_markers // num_threads if t != num_threads - 1 \
                                                   else n_markers

        for m in range(start, end):
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
                        marker_weights[m] += w

    return marker_values, marker_weights

def interpolate_grid2markers(x, y, xm, ym, grid_values, indexing="bisect", return_weights=False, ghost_nodes=True, real=np.float64):
    """
    Interpolates grid values to the markers using distance-weighted trilinear interpolation.
    
    Parameters:
    - x, y, z: 1D arrays defining the coordinates of the regular grid in 2D
    - grid_values: tuple of 2D arrays of shape (len(x), len(y)) containing grid values to interpolate
    - indexing: str, optional, default: "bisect". Indexing mode for grid nodes.
                "equidistant": grid nodes are equidistantly spaced
                "bisect": grid nodes are non-equidistantly spaced and indices are computed by bisection
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

    # Grid dimensions and grid spacing
    nx, ny = len(x), len(y)

    # Initialize marker values and weights
    n_markers = len(xm)
    q = len(grid_values)

    marker_values = tuple(np.zeros((n_markers,), dtype=real) for _ in range(q))
    marker_weights = np.zeros((n_markers,), dtype=real)

    # Get number of threads
    n_threads = nb.get_num_threads()

    # 1) Compute grid indices for each marker
    if indexing == "equidistant":
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        # xidx = ((xm - x[0]) / dx).astype(np.int32)
        # yidx = ((ym - y[0]) / dy).astype(np.int32)
        xidx = compute_idx(x, xm)
        yidx = compute_idx(y, ym)
        assert np.all(xidx >= 0) and np.all(xidx < nx), "Some markers are outside the grid."
    elif indexing == "bisect":
        xidx = bisect_idx(x, xm, nx)
        yidx = bisect_idx(y, ym, ny)
    else:
        raise ValueError("Invalid indexing mode. Choose 'equidistant' or 'bisect'.")
    
    # 2) Loop over markers and accumulate weighted values and weights
    marker_values, marker_weights = reduce_grid_values(x, y,
                                                       xm, ym,
                                                       xidx, yidx,
                                                       grid_values, marker_values,
                                                       marker_weights,
                                                       n_threads)
    
    # Assert that all weights are positive
    # assert np.all(marker_weights > 0), "Some markers have zero weight. Check grid positions."

    if return_weights:
        return marker_values, marker_weights
    else:
        rax = tuple(v / marker_weights for v in marker_values)
        return rax if len(rax) > 1 else rax[0] # Return a single value if only one quantity is interpolated
