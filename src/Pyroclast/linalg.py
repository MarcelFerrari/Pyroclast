
"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: linalg.py
Description: Linear algebra utilities for Pyroclast.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

See LICENSE file in the project root for full license information.
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
