"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/model/stokes_2D_mg/smoothers/vx/__init__.py
Description: This package contains sub implementations that only compute the vx update. (needed to different vx update schemes)

Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
from ._inline_vx import (cpu_inline_loop_body_vx,
                        cpu_prep_vx_cache,
                        cpu_compute_coeffs_vx,
                        cpu_compute_neighbor_sum_vx,
                        gpu_inline_loop_body_vx,
                        gpu_compute_coeffs_vx,
                        gpu_compute_neighbor_sum_vx)
