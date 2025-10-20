"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: gpu_utils.py
Description: Helper utilities for GPU execution parameters and interoperability.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

# Attempt to import CuPy and Numba CUDA, set to None if not available
try:
    import cupy as cp
    from numba import cuda
except ImportError:
    cp = None
    cuda = None

_block_size = (32, 8)  # (bx, by) following cuda convention (x, y)

def get_block_size():
    """Get the default block size for GPU kernels."""
    return _block_size

def get_numba_stream():
    """Get the current CuPy stream for interoperability with Numba CUDA."""
    if cp is None or cuda is None:
        raise RuntimeError("Attempted to get CuPy stream, but CuPy or Numba CUDA is not available.")
    cp_stream = cp.cuda.get_current_stream()
    ptr = int(cp_stream.ptr)
    return cuda.external_stream(ptr)


def launch_2D(shape, max_blocks=None):
    """
    Utility to determine grid/block sizes for 2D CUDA kernels.

    Parameters
    ----------
    shape : tuple[int, int]
        (ny, nx) grid shape — i.e. (rows, cols).
    max_blocks : int or None, optional
        If provided, limits total number of blocks (grid.x * grid.y)
        to this maximum — useful for cooperative launches.
        If None, the full grid is used.

    Returns
    -------
    grid : tuple[int, int]
        Grid dimensions (gridDim.x, gridDim.y)
    block : tuple[int, int]
        Block dimensions (blockDim.x, blockDim.y)
    """
    ny, nx = shape
    bx, by = _block_size  # global or external block size, e.g. (32, 4)

    # Compute full grid that covers the whole array
    gx = (nx + bx - 1) // bx
    gy = (ny + by - 1) // by

    if max_blocks is not None:
        total = gx * gy
        if total > max_blocks:
            # shrink proportionally while keeping roughly same aspect ratio
            aspect = gx / gy
            gy = int((max_blocks / aspect) ** 0.5)
            gx = max(1, max_blocks // max(1, gy))
            gy = max(1, gy)
            # ensure product doesn't exceed
            if gx * gy > max_blocks:
                gy = max(1, max_blocks // gx)

    return (gx, gy), (bx, by)
