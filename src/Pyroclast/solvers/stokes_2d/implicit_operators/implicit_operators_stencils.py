"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: implicit_operators.py
Description: This file implements the implicit operators for the Stokes flow
             and continuity equations in 2D.
Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

# implicit_operators_refactored.py
import numpy as np
import numba as nb

# ---------------------------
# CPU point kernels (always inlined)
# ---------------------------

def vx_op_point(i, j, dx, dy, etap, etab, vx, vy, p):
    # viscosities around (i, j) on staggered grid
    etaA = etap[i,   j]
    etaB = etap[i,   j+1]
    eta1 = etab[i-1, j]
    eta2 = etab[i,   j]

    # coefficients
    vx1 = 2.0 * etaA / (dx * dx)
    vx2 = eta1 / (dy * dy)
    vx3 = -(eta1 + eta2)/(dy * dy) - 2.0*(etaA + etaB)/(dx * dx)
    vx4 = eta2 / (dy * dy)
    vx5 = 2.0 * etaB / (dx * dx)

    # cross terms with vy
    vy1 =  eta1 / (dx * dy)
    vy2 = -eta2 / (dx * dy)
    vy3 = -eta1 / (dx * dy)
    vy4 =  eta2 / (dx * dy)

    # pressure gradient (staggered p on cell centers)
    dp_right = -p[i, j+1] / dx
    dp_left  = +p[i, j]   / dx

    return (
        vx1 * vx[i,   j-1] +
        vx2 * vx[i-1, j  ] +
        vx3 * vx[i,   j  ] +
        vx4 * vx[i+1, j  ] +
        vx5 * vx[i,   j+1] +
        vy1 * vy[i-1, j  ] +
        vy2 * vy[i,   j  ] +
        vy3 * vy[i-1, j+1] +
        vy4 * vy[i,   j+1] +
        dp_right + dp_left
    )


def vy_op_point(i, j, dx, dy, etap, etab, vx, vy, p):
    etaA = etap[i,   j]
    etaB = etap[i+1, j]
    eta1 = etab[i,   j-1]
    eta2 = etab[i,   j]

    vy1 = eta1 / (dx * dx)
    vy2 = 2.0 * etaA / (dy * dy)
    vy3 = (-2.0 * etaA / (dy * dy)
           -2.0 * etaB / (dy * dy)
           -eta1 / (dx * dx)
           -eta2 / (dx * dx))
    vy4 = 2.0 * etaB / (dy * dy)
    vy5 = eta2 / (dx * dx)

    # cross terms with vx
    vx1 =  eta1 / (dx * dy)
    vx2 = -eta1 / (dx * dy)
    vx3 = -eta2 / (dx * dy)
    vx4 =  eta2 / (dx * dy)

    # pressure gradient
    dp_up   = -p[i+1, j] / dy
    dp_down = +p[i,   j] / dy

    return (
        vy1 * vy[i,   j-1] +
        vy2 * vy[i-1, j  ] +
        vy3 * vy[i,   j  ] +
        vy4 * vy[i+1, j  ] +
        vy5 * vy[i,   j+1] +
        vx1 * vx[i,   j-1] +
        vx2 * vx[i+1, j-1] +
        vx3 * vx[i,   j  ] +
        vx4 * vx[i+1, j  ] +
        dp_up + dp_down
    )


def p_op_point(i, j, dx, dy, vx, vy):
    return ( (vx[i, j] - vx[i, j - 1]) / dx
           + (vy[i, j] - vy[i - 1, j]) / dy )


def uzawa_vx_op_point(i, j, dx, dy, etap, etab, vx, vy):
    etaA = etap[i,   j]
    etaB = etap[i,   j+1]
    eta1 = etab[i-1, j]
    eta2 = etab[i,   j]

    vx1 = 2.0 * etaA / (dx * dx)
    vx2 = eta1 / (dy * dy)
    vx3 = -(eta1 + eta2)/(dy * dy) - 2.0*(etaA + etaB)/(dx * dx)
    vx4 = eta2 / (dy * dy)
    vx5 = 2.0 * etaB / (dx * dx)

    vy1 =  eta1 / (dx * dy)
    vy2 = -eta2 / (dx * dy)
    vy3 = -eta1 / (dx * dy)
    vy4 =  eta2 / (dx * dy)

    return (
        vx1 * vx[i,   j-1] +
        vx2 * vx[i-1, j  ] +
        vx3 * vx[i,   j  ] +
        vx4 * vx[i+1, j  ] +
        vx5 * vx[i,   j+1] +
        vy1 * vy[i-1, j  ] +
        vy2 * vy[i,   j  ] +
        vy3 * vy[i-1, j+1] +
        vy4 * vy[i,   j+1]
    )


def uzawa_vy_op_point(i, j, dx, dy, etap, etab, vx, vy):
    etaA = etap[i,   j]
    etaB = etap[i+1, j]
    eta1 = etab[i,   j-1]
    eta2 = etab[i,   j]

    vy1 = eta1 / (dx * dx)
    vy2 = 2.0 * etaA / (dy * dy)
    vy3 = (-2.0 * etaA / (dy * dy)
           -2.0 * etaB / (dy * dy)
           -eta1 / (dx * dx)
           -eta2 / (dx * dx))
    vy4 = 2.0 * etaB / (dy * dy)
    vy5 = eta2 / (dx * dx)

    vx1 =  eta1 / (dx * dy)
    vx2 = -eta1 / (dx * dy)
    vx3 = -eta2 / (dx * dy)
    vx4 =  eta2 / (dx * dy)

    return (
        vy1 * vy[i,   j-1] +
        vy2 * vy[i-1, j  ] +
        vy3 * vy[i,   j  ] +
        vy4 * vy[i+1, j  ] +
        vy5 * vy[i,   j+1] +
        vx1 * vx[i,   j-1] +
        vx2 * vx[i+1, j-1] +
        vx3 * vx[i,   j  ] +
        vx4 * vx[i+1, j  ]
    )


def uzawa_vx_rhs_point(i, j, dx, vx_rhs, p):
    # Uzawa RHS = vx_rhs + dP/dx
    return vx_rhs[i, j] + (p[i, j+1] - p[i, j]) / dx


def uzawa_vy_rhs_point(i, j, dy, vy_rhs, p):
    # Uzawa RHS = vy_rhs + dP/dy
    return vy_rhs[i, j] + (p[i+1, j] - p[i, j]) / dy


def p_energy_point(i, j, etap, p_res):
    # simple weighted L2 at point
    return etap[i, j] * p_res[i, j] * p_res[i, j]


def vx_energy_point(i, j, dx, dy, etap, etab, vx_res):
    etaA = etap[i,   j]
    etaB = etap[i,   j+1]
    eta1 = etab[i-1, j]
    eta2 = etab[i,   j]
    denom = 2.0*(etaA + etaB)/(dx*dx) + (eta1 + eta2)/(dy*dy)
    eta_inv = 1.0/denom
    v = vx_res[i, j]
    return eta_inv * v * v


def vy_energy_point(i, j, dx, dy, etap, etab, vy_res):
    etaA = etap[i,   j]
    etaB = etap[i+1, j]
    eta1 = etab[i,   j-1]
    eta2 = etab[i,   j]
    denom = 2.0*(etaA + etaB)/(dy*dy) + (eta1 + eta2)/(dx*dx)
    eta_inv = 1.0/denom
    v = vy_res[i, j]
    return eta_inv * v * v
