"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: linalg.py
Description: Device-aware linear algebra helpers for NumPy and CuPy arrays.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

def get_xp(device: str = "cpu"):
    if device == "cpu":
        return np
    elif device == "gpu":
        if cp is None:
            raise ImportError("Failed to import cupy. Please install cupy to use GPU functionality.")
        return cp
    else:
        raise ValueError(f"Unknown device: {device}")
