
"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: linalg.py
Description: Linear algebra utilities for Pyroclast.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

# Convenience functions for switching between numpy and cupy
import numpy as np

xp = np

def np2xp():
    global xp
    xp = np

def cp2xp():
    import cupy as cp
    global xp
    xp = cp
