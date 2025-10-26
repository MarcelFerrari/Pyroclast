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
from Pyroclast.mpi import get_cart_comm, MPI

try:
    raise ImportError
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


def interpolate_markers2grid(x, y, xm, ym, vals, indexing="bisect", return_weights=False, mpi=True, real=np.float64):
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
        # We perform halo exchange to reduce
        # the contributions of neighboring ranks

        # The weights and values written to the halo regions
        # are accumulated to neighboring ranks
        # Similarly, we receive contributions from neighboring ranks
        # that are accumulated to their halo regions
        # We never touch the ghost nodes (i.e. last row/column of each rank)
        comm = get_cart_comm()
        rank = comm.Get_rank()
        gi, gj = comm.Get_coords(rank)
        py, px = comm.Get_topo()[0]  # dims (rows, cols)

        # Need to detect edge ranks
        is_top_edge = (gi == 0)
        is_bottom_edge = (gi == py - 1)
        is_left_edge = (gj == 0)
        is_right_edge = (gj == px - 1)
        
        # We only send/receive valid domain regions
        row_size = nx1 - 3
        if is_left_edge or is_right_edge:
            row_size += 1
        jmin = 1 if not is_left_edge else 0
        jmax = -2 if not is_right_edge else -1

        col_size = ny1 - 3
        if is_top_edge or is_bottom_edge:
            col_size += 1
        imin = 1 if not is_top_edge else 0
        imax = -2 if not is_bottom_edge else -1

        def shift_coords(di, dj):
            i, j = gi + di, gj + dj

            # If out of bounds, return MPI.PROC_NULL
            if i < 0 or i >= py or j < 0 or j >= px:
                return MPI.PROC_NULL

            return comm.Get_cart_rank((i, j))

        # We need to perform 8-way halo exchange
        N_rank = shift_coords(-1, 0)  # North rank coordinates
        S_rank = shift_coords(1, 0)  # South rank coordinates
        E_rank = shift_coords(0, 1)  # East rank coordinates
        W_rank = shift_coords(0, -1) # West rank coordinates
        NE_rank = shift_coords(-1, 1) # North-East rank coordinates
        NW_rank = shift_coords(-1, -1) # North-West rank coordinates
        SE_rank = shift_coords(1, 1)  # South-East rank coordinates
        SW_rank = shift_coords(1, -1) # South-West rank coordinates

        # Prepare source and destination buffers for halo exchange
        reqs = []

        # North halo
        if N_rank != MPI.PROC_NULL:
            # Need to send receive from North
            N_w_recv_buf = np.empty(nx1 - 3, dtype=real)
            N_v_recv_buf = np.empty(nx1 - 3, dtype=real)

            # Post receives
            reqs.append(comm.Irecv(N_w_recv_buf, source=N_rank, tag=0)) # 0: weights
            reqs.append(comm.Irecv(N_v_recv_buf, source=N_rank, tag=1)) # 1: values

            N_w_send_buf = grid_weights[0, jmin:jmax].copy()
            N_v_send_buf = grid_values[0, jmin:jmax].copy()

            # Post sends
            reqs.append(comm.Isend(N_w_send_buf, dest=N_rank, tag=0)) # 0: weights
            reqs.append(comm.Isend(N_v_send_buf, dest=N_rank, tag=1)) # 1: values

        # South halo
        if S_rank != MPI.PROC_NULL:
            # Need to send receive from South
            S_w_recv_buf = np.empty(nx1 - 3, dtype=real)
            S_v_recv_buf = np.empty(nx1 - 3, dtype=real)
            
            # Post receives
            reqs.append(comm.Irecv(S_w_recv_buf, source=S_rank, tag=0)) # 0: weights
            reqs.append(comm.Irecv(S_v_recv_buf, source=S_rank, tag=1)) # 1: values

            S_w_send_buf = grid_weights[-2, 1:-2]
            S_v_send_buf = grid_values[-2, 1:-2]

            # Post sends
            reqs.append(comm.Isend(S_w_send_buf, dest=S_rank, tag=0)) # 0: weights
            reqs.append(comm.Isend(S_v_send_buf, dest=S_rank, tag=1)) # 1: values

        # West halo
        if W_rank != MPI.PROC_NULL:
            # Need to send receive from West
            W_w_recv_buf = np.empty(ny1 - 3, dtype=real)
            W_v_recv_buf = np.empty(ny1 - 3, dtype=real)
            
            # Post receives
            reqs.append(comm.Irecv(W_w_recv_buf, source=W_rank, tag=0)) # 0: weights
            reqs.append(comm.Irecv(W_v_recv_buf, source=W_rank, tag=1)) # 1: values

            W_w_send_buf = grid_weights[1:-2, 0].copy()
            W_v_send_buf = grid_values[1:-2, 0].copy()

            # Post sends
            reqs.append(comm.Isend(W_w_send_buf, dest=W_rank, tag=0)) # 0: weights
            reqs.append(comm.Isend(W_v_send_buf, dest=W_rank, tag=1)) # 1: values

        # East halo
        if E_rank != MPI.PROC_NULL:
            # Need to send receive from East
            E_w_recv_buf = np.empty(ny1 - 3, dtype=real)
            E_v_recv_buf = np.empty(ny1 - 3, dtype=real)
            
            # Post receives
            reqs.append(comm.Irecv(E_w_recv_buf, source=E_rank, tag=0)) # 0: weights
            reqs.append(comm.Irecv(E_v_recv_buf, source=E_rank, tag=1)) # 1: values

            E_w_send_buf = grid_weights[1:-2, -2].copy()
            E_v_send_buf = grid_values[1:-2, -2].copy()

            # Post sends
            reqs.append(comm.Isend(E_w_send_buf, dest=E_rank, tag=0)) # 0: weights
            reqs.append(comm.Isend(E_v_send_buf, dest=E_rank, tag=1)) # 1: values

        # North-West halo
        if NW_rank != MPI.PROC_NULL:
            NW_w_recv_buf = np.empty(1, dtype=real)
            NW_v_recv_buf = np.empty(1, dtype=real)

            # Post receives
            reqs.append(comm.Irecv(NW_w_recv_buf, source=NW_rank, tag=0)) # 0: weights
            reqs.append(comm.Irecv(NW_v_recv_buf, source=NW_rank, tag=1)) # 1: values

            NW_w_send_buf = np.array([grid_weights[0, 0]], dtype=real)
            NW_v_send_buf = np.array([grid_values[0, 0]], dtype=real)

            # Post sends
            reqs.append(comm.Isend(NW_w_send_buf, dest=NW_rank, tag=0)) # 0: weights
            reqs.append(comm.Isend(NW_v_send_buf, dest=NW_rank, tag=1)) # 1: values

        # South-West halo
        if SW_rank != MPI.PROC_NULL:
            SW_w_recv_buf = np.empty(1, dtype=real)
            SW_v_recv_buf = np.empty(1, dtype=real)

            # Post receives
            reqs.append(comm.Irecv(SW_w_recv_buf, source=SW_rank, tag=0)) # 0: weights
            reqs.append(comm.Irecv(SW_v_recv_buf, source=SW_rank, tag=1)) # 1: values

            SW_w_send_buf = np.array([grid_weights[-2, 0]], dtype=real)
            SW_v_send_buf = np.array([grid_values[-2, 0]], dtype=real)

            # Post sends
            reqs.append(comm.Isend(SW_w_send_buf, dest=SW_rank, tag=0)) # 0: weights
            reqs.append(comm.Isend(SW_v_send_buf, dest=SW_rank, tag=1)) # 1: values

        # North-East halo
        if NE_rank != MPI.PROC_NULL:
            NE_w_recv_buf = np.empty(1, dtype=real)
            NE_v_recv_buf = np.empty(1, dtype=real)

            # Post receives
            reqs.append(comm.Irecv(NE_w_recv_buf, source=NE_rank, tag=0)) # 0: weights
            reqs.append(comm.Irecv(NE_v_recv_buf, source=NE_rank, tag=1)) # 1: values

            NE_w_send_buf = np.array([grid_weights[0, -2]], dtype=real)
            NE_v_send_buf = np.array([grid_values[0, -2]], dtype=real)

            # Post sends
            reqs.append(comm.Isend(NE_w_send_buf, dest=NE_rank, tag=0)) # 0: weights
            reqs.append(comm.Isend(NE_v_send_buf, dest=NE_rank, tag=1)) # 1: values

        # South-East halo
        if SE_rank != MPI.PROC_NULL:
            SE_w_recv_buf = np.empty(1, dtype=real)
            SE_v_recv_buf = np.empty(1, dtype=real)

            # Post receives
            reqs.append(comm.Irecv(SE_w_recv_buf, source=SE_rank, tag=0)) # 0: weights
            reqs.append(comm.Irecv(SE_v_recv_buf, source=SE_rank, tag=1)) # 1: values

            SE_w_send_buf = np.array([grid_weights[-2, -2]], dtype=real)
            SE_v_send_buf = np.array([grid_values[-2, -2]], dtype=real)

            # Post sends
            reqs.append(comm.Isend(SE_w_send_buf, dest=SE_rank, tag=0)) # 0: weights
            reqs.append(comm.Isend(SE_v_send_buf, dest=SE_rank, tag=1)) # 1: values

        # Wait for all communications to complete
        MPI.Request.Waitall(reqs)

        # Accumulate received halo contributions
        # Remember that we need to write to our own domain
        # regions, not the halo regions!
        # We will update halos once the full grid_values
        # are normalized
        if N_rank != MPI.PROC_NULL:
            grid_weights[1, 1:-2] += N_w_recv_buf
            grid_values[1, 1:-2] += N_v_recv_buf

        if S_rank != MPI.PROC_NULL:
            grid_weights[-3, 1:-2] += S_w_recv_buf
            grid_values[-3, 1:-2] += S_v_recv_buf
        
        if W_rank != MPI.PROC_NULL:
            grid_weights[1:-2, 1] += W_w_recv_buf
            grid_values[1:-2, 1] += W_v_recv_buf

        if E_rank != MPI.PROC_NULL:
            grid_weights[1:-2, -3] += E_w_recv_buf
            grid_values[1:-2, -3] += E_v_recv_buf

        if NW_rank != MPI.PROC_NULL:
            grid_weights[1, 1] += NW_w_recv_buf
            grid_values[1, 1] += NW_v_recv_buf

        if SW_rank != MPI.PROC_NULL:
            grid_weights[-3, 1] += SW_w_recv_buf
            grid_values[-3, 1] += SW_v_recv_buf

        if NE_rank != MPI.PROC_NULL:
            grid_weights[1, -3] += NE_w_recv_buf
            grid_values[1, -3] += NE_v_recv_buf

        if SE_rank != MPI.PROC_NULL:
            grid_weights[-3, -3] += SE_w_recv_buf
            grid_values[-3, -3] += SE_v_recv_buf

    
    # Normalize grid values by weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore division by zero!
        # This is correct and represents values outside the grid!
        grid_values /= grid_weights

    # Halo-exchange of normalized grid values if necessary
    if mpi:
        # We have correctly composed the grid values, including the
        # boundaries of the domain that are affected by particles from
        # neighboring ranks. However, the halo regions of each rank are still
        # not updated and only store remote writes to the boundaries of neighboring ranks.
        # We need to perform a final halo exchange of the normalized grid values to
        # ensure that each rank has the correct values in its halo regions.
        comm = get_cart_comm()

        # We need to perform 8-way halo exchange
        N_rank = shift_coords(-1, 0)  # North rank coordinates
        S_rank = shift_coords(1, 0)  # South rank coordinates
        E_rank = shift_coords(0, 1)  # East rank coordinates
        W_rank = shift_coords(0, -1) # West rank coordinates
        NE_rank = shift_coords(-1, 1) # North-East rank coordinates
        NW_rank = shift_coords(-1, -1) # North-West rank coordinates
        SE_rank = shift_coords(1, 1)  # South-East rank coordinates
        SW_rank = shift_coords(1, -1) # South-West rank coordinates

        # Prepare source and destination buffers for halo exchange
        reqs = []

        # North halo
        if N_rank != MPI.PROC_NULL:
            # Post receives
            N_recv_buf = np.empty(nx1 - 3, dtype=real)
            reqs.append(comm.Irecv(N_recv_buf, source=N_rank))

            # Post sends
            N_send_buf = grid_values[1, 1:-2]
            reqs.append(comm.Isend(N_send_buf, dest=N_rank))

        # South halo
        if S_rank != MPI.PROC_NULL:
            # Post receives
            S_recv_buf = np.empty(nx1 - 3, dtype=real)
            reqs.append(comm.Irecv(S_recv_buf, source=S_rank))

            # Post sends
            S_send_buf = grid_values[-3, 1:-2]
            reqs.append(comm.Isend(S_send_buf, dest=S_rank))

        # West halo
        if W_rank != MPI.PROC_NULL:
            # Post receives
            W_recv_buf = np.empty(ny1 - 3, dtype=real)
            reqs.append(comm.Irecv(W_recv_buf, source=W_rank))

            # Post sends
            W_send_buf = grid_values[1:-2, 1].copy()
            reqs.append(comm.Isend(W_send_buf, dest=W_rank))

        # East halo
        if E_rank != MPI.PROC_NULL:
            # Post receives
            E_recv_buf = np.empty(ny1 - 3, dtype=real)
            reqs.append(comm.Irecv(E_recv_buf, source=E_rank))

            # Post sends
            E_send_buf = grid_values[1:-2, -3].copy()
            reqs.append(comm.Isend(E_send_buf, dest=E_rank))

        # North-West halo
        if NW_rank != MPI.PROC_NULL:
            # Post receives
            NW_recv_buf = np.empty(1, dtype=real)
            reqs.append(comm.Irecv(NW_recv_buf, source=NW_rank))

            # Post sends
            NW_send_buf = np.array([grid_values[1, 1]], dtype=real)
            reqs.append(comm.Isend(NW_send_buf, dest=NW_rank))
        
        # South-West halo
        if SW_rank != MPI.PROC_NULL:
            # Post receives
            SW_recv_buf = np.empty(1, dtype=real)
            reqs.append(comm.Irecv(SW_recv_buf, source=SW_rank))

            # Post sends
            SW_send_buf = np.array([grid_values[-3, 1]], dtype=real)
            reqs.append(comm.Isend(SW_send_buf, dest=SW_rank))

        # North-East halo
        if NE_rank != MPI.PROC_NULL:
            # Post receives
            NE_recv_buf = np.empty(1, dtype=real)
            reqs.append(comm.Irecv(NE_recv_buf, source=NE_rank))

            # Post sends
            NE_send_buf = np.array([grid_values[1, -3]], dtype=real)
            reqs.append(comm.Isend(NE_send_buf, dest=NE_rank))

        # South-East halo
        if SE_rank != MPI.PROC_NULL:
            # Post receives
            SE_recv_buf = np.empty(1, dtype=real)
            reqs.append(comm.Irecv(SE_recv_buf, source=SE_rank))

            # Post sends
            SE_send_buf = np.array([grid_values[-3, -3]], dtype=real)
            reqs.append(comm.Isend(SE_send_buf, dest=SE_rank))

        # Wait for all communications to complete
        MPI.Request.Waitall(reqs)

        # Update halo regions with received data
        if N_rank != MPI.PROC_NULL:
            grid_values[0, 1:-2] = N_recv_buf
        
        if S_rank != MPI.PROC_NULL:
            grid_values[-2, 1:-2] = S_recv_buf

        if W_rank != MPI.PROC_NULL:
            grid_values[1:-2, 0] = W_recv_buf

        if E_rank != MPI.PROC_NULL:
            grid_values[1:-2, -2] = E_recv_buf

        if NW_rank != MPI.PROC_NULL:
            grid_values[0, 0] = NW_recv_buf

        if SW_rank != MPI.PROC_NULL:
            grid_values[-2, 0] = SW_recv_buf

        if NE_rank != MPI.PROC_NULL:
            grid_values[0, -2] = NE_recv_buf
        
        if SE_rank != MPI.PROC_NULL:
            grid_values[-2, -2] = SE_recv_buf
    
    # All done!
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