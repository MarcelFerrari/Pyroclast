"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: solvers/stokes_2d/smoothers/coeff.py
Description: Coefficient helpers for Stokes smoothing stencils.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numba as nb
import numpy as np

# -----------------------------
# Coefficient helpers (inlined)
# -----------------------------
def x_momentum_coefficients(dx, dy, etaA, etaB, eta1, eta2):
    # vx-stencil coeffs
    vx1 = 2.0 * etaA / (dx * dx)
    vx2 = eta1       / (dy * dy)
    vx3 = -(eta1 + eta2) / (dy * dy) - 2.0 * (etaA + etaB) / (dx * dx)  # diag
    vx4 = eta2 / (dy * dy)
    vx5 = 2.0 * etaB / (dx * dx)
    # cross (vy) terms
    vy1 =  eta1 / (dx * dy)
    vy2 = -eta2 / (dx * dy)
    vy3 = -eta1 / (dx * dy)
    vy4 =  eta2 / (dx * dy)
    return vx1, vx2, vx3, vx4, vx5, vy1, vy2, vy3, vy4


def y_momentum_coefficients(dx, dy, etaA, etaB, eta1, eta2):
    # vy-stencil coeffs
    vy1 = eta1 / (dx * dx)
    vy2 = 2.0 * etaA / (dy * dy)
    vy3 = -2.0 * etaA/(dy*dy) - 2.0 * etaB/(dy*dy) - eta1/(dx*dx) - eta2/(dx*dx)  # diag
    vy4 = 2.0 * etaB / (dy * dy)
    vy5 = eta2 / (dx * dx)
    # cross (vx) terms
    vx1 =  eta1 / (dx * dy)
    vx2 = -eta1 / (dx * dy)
    vx3 = -eta2 / (dx * dy)
    vx4 =  eta2 / (dx * dy)
    return vy1, vy2, vy3, vy4, vy5, vx1, vx2, vx3, vx4
