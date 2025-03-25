"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: utils.py
Description: This file implements utility functions for interpolation.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numba as nb
import numpy as np

# Binary search function for a single marker
@nb.njit(cache=True)
def binary_search(x, marker, nx):
    """
    Find the index of the node in x that is immediately to the left of the marker.
    
    Parameters:
    x (np.array): Sorted array of grid coordinates.
    marker (float): Marker coordinate.
    nx (int): Number of grid points.
    
    Returns:
    int: Index of the node in x that is immediately to the left of marker.
    """
    left = 0
    right = nx - 1
    
    while left <= right:
        mid = (left + right) // 2
        if x[mid] <= marker:
            left = mid + 1
        else:
            right = mid - 1
    
    return right  # The largest index where x[index] <= marker

# Main parallelized function to find indices for all markers
@nb.njit(cache=True, parallel=True)
def bisect_idx(x, xm, nx):
    """
    Compute the index of a point in a non-uniform grid by bisection.
    
    Parameters:
    x (np.array): Sorted array of grid coordinates.
    xm (np.array): Array of marker coordinates.
    
    Returns:
    np.array: Array of indices where idx[i] is the index of the node in x
              that is immediately to the left of xm[i].
    """
    idx = np.zeros_like(xm, dtype=np.int32)
    nm = len(xm)
    # Parallel loop over each marker
    for i in nb.prange(nm):
        idx[i] = min(max(binary_search(x, xm[i], nx), 0), nx - 2)
    
    return idx

@nb.njit(cache=True)
def clip(x, xmin, xmax):
    if x < xmin:
        return xmin
    elif x > xmax:
        return xmax
    else:
        return x

@nb.njit(parallel=True, cache=True)
def compute_idx(x, xm):
    """
    Computes the index of the reference grid node for each marker.
    
    Parameters:
    - x: 1D array defining the coordinates of the regular grid
    - xm: 1D array of shape (n_markers,) containing marker coordinates
    - nx: int. Number of grid nodes in the x-direction
    
    Returns:
    - idx: 1D array of shape (n_markers,) containing the index of the reference node for each marker
    """
    nm = len(xm)
    idx = np.empty((nm, ), dtype=np.int32)
    dx = x[1] - x[0]
    nx = len(x)
    for m in nb.prange(nm):
        idx[m] = clip(int((xm[m] - x[0]) / dx), 0, nx - 2)
    return idx