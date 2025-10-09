"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: utils.py
Description: Utility functions for Pyroclast.

Author: Marcel Ferrari, Alexander Sotoudeh
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import inspect
from functools import wraps

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


def inject_threads(func):
    """
    Decorator that:
    - Validates that `func` has a parameter named `th` (of type int in annotation).
    - At call time, injects `th=nb.get_num_threads()`.
    - Raises an Error, if
    """
    # --- Static validation at decoration time ---
    sig = inspect.signature(func)

    # Check that 'th' exists as a parameter
    if 'th' not in sig.parameters:
        raise TypeError(f"Function '{func.__name__}' must have an argument named 'th'.")

    # Check that it is annotated as int (if annotated)
    ann = sig.parameters['th'].annotation
    if ann is not inspect._empty and ann is not int:
        raise TypeError(f"Function '{func.__name__}' must annotate 'th' as type int, not {ann!r}.")

    # --- Runtime wrapper ---
    @wraps(func)
    def wrapper(*args, **kwargs):
        assert "th" not in kwargs, "ERROR, Function has thread number already injected."
        kwargs['th'] = nb.get_num_threads()
        return func(*args, **kwargs)

    return wrapper
