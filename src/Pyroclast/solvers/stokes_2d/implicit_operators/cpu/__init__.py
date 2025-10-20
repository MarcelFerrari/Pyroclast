"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: solvers/stokes_2d/implicit_operators/cpu/__init__.py
Description: CPU implicit operator exports for Stokes solvers.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from .implicit_operators import (
    # ========= Operators =========
    vx_operator,
    vx_residual,
    vy_operator,
    vy_residual,
    p_operator,
    p_residual,
    uzawa_velocity_rhs,
    uzawa_vx_operator,
    uzawa_vx_residual,
    uzawa_vy_operator,
    uzawa_vy_residual,

    # ========= Energy norms =========
    compute_p_energy_norm,
    compute_vx_energy_norm,
    compute_vy_energy_norm,
)
