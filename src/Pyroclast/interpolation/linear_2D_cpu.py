"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: 2D_linear_cpu.py
Description: This file implements 2D linear interpolation functions for CPU.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import warnings
import numba as nb
import numpy as np
from Pyroclast.profiling import timer

from Pyroclast.interpolation.utils import bisect_idx, compute_idx
from Pyroclast.mpi import get_cart_comm, MPI, halo_exchange_2D

try:
    from Pyroclast.turbo.interpolation import reduce_marker_values_2D as reduce_parallel
except ImportError:
    reduce_parallel = None

@nb.njit(cache=True)
def reduce_marker_values(nx1, ny1, x, y, n_markers, xm, ym, xidx, yidx, vals, grid_values, grid_weights):
    """
    Loops over each marker and computes the weighted sum of quantities for the surrounding grid nodes.

    Parameters:
    - x, y, z: 1D arrays defining the coordinates of the regular grid in 3D
    - xm, ym: 1D arrays of shape (n_markers,) containing marker coordinates 
    - xidx, yidx: 1D arrays of shape (n_markers,) containing the index of the reference node for each marker
    - vals: tuple of 1D arrays of shape (n_markers,) containing marker values to interpolate
    - grid_values: 2d array (len(x), len(y))
    - grid_weights: 2D zeros array (n_threads, len(x), len(y)).

    Returns:
    - grid_values: 2d array (len(x), len(y)) with interpolated values at grid nodes
    - grid_weights: 2D array (len(x), len(y)) with accumulated weights at grid nodes
    """

    # Get dimensions of tensors
    dx, dy = x[1] - x[0], y[1] - y[0]
        
    for m in range(n_markers):
        # Read marker coordinates and reference node indices
        mx, my = xm[m], ym[m]
        mj, mi = xidx[m], yidx[m]

        # Calculate distances from marker to reference node
        rx = np.abs(mx - x[mj]) / dx
        ry = np.abs(my - y[mi]) / dy

        w00 = (1 - rx) * (1 - ry)
        w10 = (1 - rx) * ry
        w01 = rx * (1 - ry)
        w11 = rx * ry

        grid_values[mi, mj] += w00 * vals[m]
        grid_values[mi+1, mj] += w10 * vals[m]
        grid_values[mi, mj+1] += w01 * vals[m]
        grid_values[mi+1, mj+1] += w11 * vals[m]

        grid_weights[mi, mj] += w00
        grid_weights[mi+1, mj] += w10
        grid_weights[mi, mj+1] += w01
        grid_weights[mi+1, mj+1] += w11
    
    return grid_values, grid_weights


def interpolate_markers2grid(x, y, xm, ym, vals, indexing="bisect", return_weights=False, mpi=False, real=np.float64):
    """
    Interpolates marker values to the grid nodes using distance-weighted linear interpolation.
    
    Parameters:
    - x, y: 1D arrays defining the coordinates of the regular grid in 2D
    - xm, ym: 1D arrays of shape (n_markers,) containing marker coordinates
    - vals: 1D array of shape (n_markers,) containing marker values to interpolate
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
    assert isinstance(vals, np.ndarray)
    
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
    nx1, ny1 = len(x), len(y)
    n_markers = len(xm)
    
    # Initialize grid_values and grid_weights
    # Allocate leading dimension for number of threads
    grid_values = np.zeros((ny1, nx1), dtype=real)
    grid_weights = np.zeros((ny1, nx1), dtype=real)

    # 1) Compute grid indices for each marker
    # Important to pass nx and ny to the indexing functions
    # in order to handle ghost nodes correctly
    if indexing == "equidistant":
        xidx = compute_idx(x, xm)
        yidx = compute_idx(y, ym)
    elif indexing == "bisect":
        xidx = bisect_idx(x, xm, nx1)
        yidx = bisect_idx(y, ym, ny1)
    else:
        raise ValueError("Invalid indexing mode. Choose 'equidistant' or 'bisect'.")

    # 2) Loop over markers and accumulate weighted values and weights
    reduction_fn = reduce_parallel if reduce_parallel is not None else reduce_marker_values
    reduction_fn(nx1, ny1, x, y, n_markers, xm, ym, xidx, yidx, vals, grid_values, grid_weights)

    # 3) Check if we are in MPI mode
    if mpi:
        # We skip ghost nodes for MPI exchange
        src_v = (
            grid_values[0, :-1],        # up
            grid_values[-2, :-1],       # down
            grid_values[:-1, 0],        # left
            grid_values[:-1, -2],       # right
            grid_values[0, 0],          # nw
            grid_values[0, -2],         # ne
            grid_values[-2, 0],         # sw
            grid_values[-2, -2]         # se
        )

        dst_v = (
            np.zeros(nx1-1, dtype=real),    # up
            np.zeros(nx1-1, dtype=real),    # down
            np.zeros(ny1-1, dtype=real),    # left
            np.zeros(ny1-1, dtype=real),    # right
            np.zeros(1, dtype=real),      # nw
            np.zeros(1, dtype=real),      # ne
            np.zeros(1, dtype=real),      # sw
            np.zeros(1, dtype=real),      # se
        )

        src_w = (
            grid_weights[0, :-1],          # up
            grid_weights[-2, :-1],         # down
            grid_weights[:-1, 0],         # left
            grid_weights[:-1, -2],         # right
            grid_weights[0, 0],          # nw
            grid_weights[0, -2],         # ne
            grid_weights[-2, 0],         # sw
            grid_weights[-2, -2],        # se
        )

        dst_w = (
            np.zeros(nx1-1, dtype=real),    # up
            np.zeros(nx1-1, dtype=real),    # down
            np.zeros(ny1-1, dtype=real),    # left
            np.zeros(ny1-1, dtype=real),    # right
            np.zeros(1, dtype=real),      # nw
            np.zeros(1, dtype=real),      # ne
            np.zeros(1, dtype=real),      # sw
            np.zeros(1, dtype=real),      # se
        )


        reqs = halo_exchange_2D(dst_v, src_v, wait=False)
        reqs += halo_exchange_2D(dst_w, src_w, wait=False)

        # Wait for all communications to complete
        MPI.Request.Waitall(reqs)

        # Reduce received halo values into local grid
        # Values
        grid_values[0, :-1] += dst_v[0]        # up
        grid_values[-2, :-1] += dst_v[1]       # down
        grid_values[:-1, 0] += dst_v[2]        # left
        grid_values[:-1, -2] += dst_v[3]       # right
        grid_values[0, 0] += dst_v[4]          # nw
        grid_values[0, -2] += dst_v[5]         # ne
        grid_values[-2, 0] += dst_v[6]         # sw
        grid_values[-2, -2] += dst_v[7]        # se

        # Weights
        grid_weights[0, :-1] += dst_w[0]        # up
        grid_weights[-2, :-1] += dst_w[1]       # down
        grid_weights[:-1, 0] += dst_w[2]        # left
        grid_weights[:-1, -2] += dst_w[3]       # right
        grid_weights[0, 0] += dst_w[4]          # nw
        grid_weights[0, -2] += dst_w[5]         # ne
        grid_weights[-2, 0] += dst_w[6]         # sw
        grid_weights[-2, -2] += dst_w[7]        # se

    if return_weights: # We are done
        return grid_values, grid_weights
    else: # Normalize grid values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Ignore division by zero!
            # This is correct and represents values outside the grid!
            grid_values /= grid_weights
        return grid_values

@nb.njit(parallel=True, cache=True)
def reduce_grid_values(x, y, xm, ym, xidx, yidx, grid_values, marker_values):
    """
    Reduces the grid values to the markers using distance-weighted linear interpolation.

    Parameters:
    - x, y: 1D arrays defining the coordinates of the regular grid in 2D
    - xm, ym: 1D arrays of shape (n_markers,) containing marker coordinates
    - xidx, yidx: 1D arrays of shape (n_markers,) containing the index of the reference node for each marker
    - grid_values: 2D array of shape (len(y), len(x)) containing grid values to interpolate
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
                    marker_values[m] += w * grid_values[gy, gx]
    return marker_values


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
    - marker_values: 1D array of shape (n_markers,) with normalized interpolated values at markers
    If return_weights is False:
    - marker_values: tuple of 1D arrays of shape (n_markers,) with non-normalized interpolated values at markers
    - marker_weights: 1D array of shape (n_markers,) containing accumulated weights at each marker
    """
    
    # Assert that input arrays are CuPy arrays and have dtype = real
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(xm, np.ndarray)
    assert isinstance(ym, np.ndarray)
    assert isinstance(grid_values, np.ndarray)

    # Grid dimensions and grid spacing
    nx, ny = len(x), len(y)

    # Initialize marker values and weights
    n_markers = len(xm)
    marker_values = np.zeros((n_markers,), dtype=real)

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
    
    # Unpack the marker_values tuple if only one quantity is interpolated
    return marker_values