"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: rng.py
Description: Random number generation utilities for Pyroclast, including setting seeds for reproducibility.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


import numpy as np
import random
import numba as nb

def set_seed(seed):
   # Set random seed for reproducibility
   random.seed(seed)
   np.random.seed(seed)
   _set_numba_seed(seed)

@nb.njit(cache=True)
def _set_numba_seed(seed):
   np.random.seed(seed)

