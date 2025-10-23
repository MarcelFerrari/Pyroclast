"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: utils.py
Description: Utility functions for Pyroclast.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numba as nb
@nb.njit(cache=True, inline='always')
def clip(x, xmin, xmax):
    """
    Clip a value x to the range [a, b].
    """
    if x < xmin:
        return xmin
    elif x > xmax:
        return xmax
    else:
        return x
    
@nb.njit(cache = True, parallel = True)
def wrap_periodic(x, xmin, xmax):
    """
    Wrap a 1D array x to the periodic range [xmin, xmax).
    """
    range_size = xmax - xmin
    for i in nb.prange(x.shape[0]):
        while x[i] < xmin:
            x[i] += range_size
        while x[i] >= xmax:
            x[i] -= range_size
    return x
