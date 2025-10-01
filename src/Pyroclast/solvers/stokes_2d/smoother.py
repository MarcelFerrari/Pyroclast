"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: smoother.py
Description: This file implements Uzawa smoother for saddle-point pressure-velocity systems.
             
Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numba as nb
import numpy as np
from .bc import apply_p_BC, cpu_apply_vx_BC, cpu_apply_vy_BC

@nb.njit(cache=True, parallel=True)
def pressure_sweep(nx1, ny1, dx, dy,
                   vx, vy, p,
                   beta,
                   relax_p, rhs):
    
    # 1) Update only interior cells
    for i in nb.prange(1, ny1 - 1):
        for j in nb.prange(1, nx1 - 1):
            # The continuity residual
            res = rhs[i, j] - ((vx[i, j] - vx[i, j - 1])/dx +
                               (vy[i, j] - vy[i - 1, j])/dy)

            # Point-wise update of pressure
            p[i, j] += res * beta[i, j] * relax_p

    # Ensure zero-mean
    pbar = np.mean(p[1:-1, 1:-1])
    p -= pbar

    # Apply pressure boundary conditions
    apply_p_BC(p)

    return p

# -----------------------------
# Coefficient helpers (inlined)
# -----------------------------
@nb.njit(inline='always')
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


@nb.njit(inline='always')
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

@nb.njit(cache=True, parallel=True)
def _vx_rb_gs_sweep(nx1, ny1,
                    dx, dy,
                    etap, etab,
                    vx, vy,
                    relax_v, rhs, BC):
    """
    In-place Red-Black Gauss-Seidel update for vx.
    """

    #----------------------------
    #  Red pass: (i + j) % 2 == 0
    #----------------------------
    for i in nb.prange(1, ny1 - 1):
        j_start = 1 if i % 2 == 0 else 2  # Red pass starts on even (i+j)
        for j in range(j_start, nx1 - 2, 2):
            # 1) Gather local viscosities
            etaA = etap[i,   j]
            etaB = etap[i,   j+1]
            eta1 = etab[i-1, j]
            eta2 = etab[i,   j]

            
            # 2) Construct coefficients for x-momentum
            vx1_coeff = 2.0 * etaA / (dx * dx)
            vx2_coeff = eta1     / (dy * dy)
            vx3_coeff = -(eta1 + eta2) / (dy * dy) \
                        - 2.0*(etaA + etaB)/(dx * dx)
            vx4_coeff = eta2 / (dy * dy)
            vx5_coeff = 2.0 * etaB / (dx * dx)

            # Cross terms with vy
            vy1_coeff =  eta1 / (dx * dy)
            vy2_coeff = -eta2 / (dx * dy)
            vy3_coeff = -eta1 / (dx * dy)
            vy4_coeff =  eta2 / (dx * dy)
            
            # 3) Sum neighbor contributions
            sum_neighbors = (
                vx1_coeff * vx[i,   j-1] +
                vx2_coeff * vx[i-1, j  ] +
                vx4_coeff * vx[i+1, j  ] +
                vx5_coeff * vx[i,   j+1]
                +
                vy1_coeff * vy[i-1, j  ] +
                vy2_coeff * vy[i,   j  ] +
                vy3_coeff * vy[i-1, j+1] +
                vy4_coeff * vy[i,   j+1]
            )

            diag = vx3_coeff

            # Gauss-Seidel in-place update
            vx[i, j] = (1.0 - relax_v)*vx[i, j] \
                        + relax_v*(rhs[i, j] - sum_neighbors)/diag
    # Apply vx boundary conditions
    cpu_apply_vx_BC(vx, BC)
    
    #----------------------------
    #  Black pass: (i + j) % 2 == 1
    #----------------------------
    for i in nb.prange(1, ny1 - 1):
        j_start = 2 if i % 2 == 0 else 1  # Black pass starts on odd (i+j)
        for j in range(j_start, nx1 - 2, 2):                
                # 1) Gather local viscosities
                etaA = etap[i,   j]
                etaB = etap[i,   j+1]
                eta1 = etab[i-1, j]
                eta2 = etab[i,   j]

                # 2) Construct coefficients for x-momentum
                vx1_coeff = 2.0 * etaA / (dx * dx)
                vx2_coeff = eta1     / (dy * dy)
                vx3_coeff = -(eta1 + eta2) / (dy * dy) \
                            - 2.0*(etaA + etaB)/(dx * dx)
                vx4_coeff = eta2 / (dy * dy)
                vx5_coeff = 2.0 * etaB / (dx * dx)

                # Cross terms with vy
                vy1_coeff =  eta1 / (dx * dy)
                vy2_coeff = -eta2 / (dx * dy)
                vy3_coeff = -eta1 / (dx * dy)
                vy4_coeff =  eta2 / (dx * dy)

                # 3) Sum neighbor contributions
                
                sum_neighbors = (
                    vx1_coeff * vx[i,   j-1] +
                    vx2_coeff * vx[i-1, j  ] +
                    vx4_coeff * vx[i+1, j  ] +
                    vx5_coeff * vx[i,   j+1]
                    +
                    vy1_coeff * vy[i-1, j  ] +
                    vy2_coeff * vy[i,   j  ] +
                    vy3_coeff * vy[i-1, j+1] +
                    vy4_coeff * vy[i,   j+1]
                )

                diag = vx3_coeff

                # Gauss-Seidel in-place update
                vx[i, j] = (1.0 - relax_v)*vx[i, j] \
                           + relax_v*(rhs[i, j] - sum_neighbors)/diag
    
    # Apply vx boundary conditions
    cpu_apply_vx_BC(vx, BC)

    return vx

@nb.njit(cache=True, parallel=True)
def _vy_red_black_gs_sweep(nx1, ny1,
                           dx, dy,
                           etap, etab,
                           vx, vy,
                           relax_v, rhs, BC):
    """
    In-place Red-Black Gauss-Seidel update for vy.
    """

    #----------------------------
    #  Red pass
    #----------------------------
    for i in nb.prange(1, ny1 - 2):
        for j in range(1, nx1 - 1):
            if (i + j) % 2 == 0:
                # 1) Gather local viscosities
                etaA = etap[i,   j]
                etaB = etap[i+1, j]
                eta1 = etab[i,   j-1]
                eta2 = etab[i,   j]

                # 2) Construct coefficients for y-momentum                
                vy1_coeff = eta1 / (dx * dx)
                vy2_coeff = 2.0 * etaA / (dy * dy)
                vy3_coeff = -2.0 * etaA/(dy*dy) \
                            -2.0 * etaB/(dy*dy) \
                            - eta1/(dx*dx) \
                            - eta2/(dx*dx)
                vy4_coeff = 2.0 * etaB / (dy * dy)
                vy5_coeff = eta2 / (dx * dx)

                # Cross terms with vx
                vx1_coeff =  eta1 / (dx * dy)
                vx2_coeff = -eta1 / (dx * dy)
                vx3_coeff = -eta2 / (dx * dy)
                vx4_coeff =  eta2 / (dx * dy)

                
                # 3) Sum neighbor contributions
                sum_neighbors = (
                    # vy neighbors
                    vy1_coeff * vy[i,   j-1] +
                    vy2_coeff * vy[i-1, j  ] +
                    vy4_coeff * vy[i+1, j  ] +
                    vy5_coeff * vy[i,   j+1]
                    +
                    # cross terms with vx
                    vx1_coeff * vx[i,   j-1] +
                    vx2_coeff * vx[i+1, j-1] +
                    vx3_coeff * vx[i,   j  ] +
                    vx4_coeff * vx[i+1, j  ]
                )

                diag = vy3_coeff
            
                # 4) Gauss-Seidel in-place update
                vy[i, j] = (1.0 - relax_v)*vy[i, j] \
                           + relax_v*(rhs[i, j] - sum_neighbors)/diag
    # Apply vy boundary conditions
    cpu_apply_vy_BC(vy, BC)

    #----------------------------
    #  Black pass
    #----------------------------
    for i in nb.prange(1, ny1 - 2):
        for j in range(1, nx1 - 1):
            if (i + j) % 2 == 1:
                # 1) Gather viscosities
                etaA = etap[i,   j]
                etaB = etap[i+1, j]
                eta1 = etab[i,   j-1]
                eta2 = etab[i,   j]

                # 2) Coefficients for y-momentum                
                vy1_coeff = eta1 / (dx * dx)
                vy2_coeff = 2.0 * etaA / (dy * dy)
                vy3_coeff = -2.0 * etaA/(dy*dy) \
                            -2.0 * etaB/(dy*dy) \
                            - eta1/(dx*dx) \
                            - eta2/(dx*dx)
                vy4_coeff = 2.0 * etaB / (dy * dy)
                vy5_coeff = eta2 / (dx * dx)

                # Cross terms with vx
                vx1_coeff =  eta1 / (dx * dy)
                vx2_coeff = -eta1 / (dx * dy)
                vx3_coeff = -eta2 / (dx * dy)
                vx4_coeff =  eta2 / (dx * dy)

                # 3) Sum neighbors                
                sum_neighbors = (
                    vy1_coeff * vy[i,   j-1] +
                    vy2_coeff * vy[i-1, j  ] +
                    vy4_coeff * vy[i+1, j  ] +
                    vy5_coeff * vy[i,   j+1]
                    +
                    vx1_coeff * vx[i,   j-1] +
                    vx2_coeff * vx[i+1, j-1] +
                    vx3_coeff * vx[i,   j  ] +
                    vx4_coeff * vx[i+1, j  ]
                )

                diag = vy3_coeff

                # 4) In-place update                
                vy[i, j] = (1.0 - relax_v)*vy[i, j] \
                           + relax_v*(rhs[i, j] - sum_neighbors)/diag
    
    # Apply vy boundary conditions
    cpu_apply_vy_BC(vy, BC)

    return vy

@nb.njit(cache=True)
def velocity_smoother(nx1, ny1,
                      dx, dy,
                      etap, etab,
                      vx, vy,
                      relax_v, BC,
                      vx_rhs, vy_rhs, max_iter):
    """
    Full Uzawa smoother for velocity and pressure.
    """
    for _ in range(max_iter):
        vx = _vx_rb_gs_sweep(nx1, ny1,
                            dx, dy,
                            etap, etab,
                            vx, vy,
                            relax_v, vx_rhs, BC)
        
        vy = _vy_red_black_gs_sweep(nx1, ny1,
                                    dx, dy,
                                    etap, etab,
                                    vx, vy,
                                    relax_v, vy_rhs, BC)
        
    return vx, vy

@nb.njit(cache=True, parallel=True)
def _vx_jacobi_sweep(nx1, ny1,
                     dx, dy,
                     etap, etab,
                     vx_old, vy_old,
                     relax_v, rhs,
                     vx_new_out):
    """
    One Jacobi sweep for vx.
    Reads from (vx_old, vy_old), writes into vx_new_out.
    Interior ranges mirror your GS stencil: i in [1..ny1-2], j in [1..nx1-3].
    """
    for i in nb.prange(1, ny1 - 1):        # i: 1 .. ny1-2
        for j in range(1, nx1 - 2):        # j: 1 .. nx1-3
            etaA = etap[i,   j]
            etaB = etap[i,   j+1]
            eta1 = etab[i-1, j]
            eta2 = etab[i,   j]

            vx1, vx2, vx3, vx4, vx5, vy1, vy2, vy3, vy4 = x_momentum_coefficients(
                dx, dy, etaA, etaB, eta1, eta2
            )

            sum_neighbors = (
                vx1 * vx_old[i,   j-1] +
                vx2 * vx_old[i-1, j  ] +
                vx4 * vx_old[i+1, j  ] +
                vx5 * vx_old[i,   j+1]
                +
                vy1 * vy_old[i-1, j  ] +
                vy2 * vy_old[i,   j  ] +
                vy3 * vy_old[i-1, j+1] +
                vy4 * vy_old[i,   j+1]
            )

            vx_new_out[i, j] = (1.0 - relax_v) * vx_old[i, j] + relax_v * (rhs[i, j] - sum_neighbors) / vx3

    return vx_new_out


@nb.njit(cache=True, parallel=True)
def _vy_jacobi_sweep(nx1, ny1,
                     dx, dy,
                     etap, etab,
                     vx_old, vy_old,
                     relax_v, rhs,
                     vy_new_out):
    """
    One Jacobi sweep for vy.
    Reads from (vx_old, vy_old), writes into vy_new_out.
    Interior ranges mirror your GS stencil: i in [1..ny1-3], j in [1..nx1-2].
    """
    for i in nb.prange(1, ny1 - 2):        # i: 1 .. ny1-3
        for j in range(1, nx1 - 1):        # j: 1 .. nx1-2
            etaA = etap[i,   j]
            etaB = etap[i+1, j]
            eta1 = etab[i,   j-1]
            eta2 = etab[i,   j]

            vy1, vy2, vy3, vy4, vy5, vx1, vx2, vx3, vx4 = y_momentum_coefficients(
                dx, dy, etaA, etaB, eta1, eta2
            )

            sum_neighbors = (
                vy1 * vy_old[i,   j-1] +
                vy2 * vy_old[i-1, j  ] +
                vy4 * vy_old[i+1, j  ] +
                vy5 * vy_old[i,   j+1]
                +
                vx1 * vx_old[i,   j-1] +
                vx2 * vx_old[i+1, j-1] +
                vx3 * vx_old[i,   j  ] +
                vx4 * vx_old[i+1, j  ]
            )

            vy_new_out[i, j] = (1.0 - relax_v) * vy_old[i, j] + relax_v * (rhs[i, j] - sum_neighbors) / vy3

    return vy_new_out


@nb.njit(cache=True)
def velocity_jacobi_smoother(nx1, ny1,
                             dx, dy,
                             etap, etab,
                             vx, vy,              # initial guesses (also serve as buffers)
                             vx_new, vy_new,      # alternate preallocated buffers
                             relax_v, BC,
                             vx_rhs, vy_rhs,
                             max_iter):
    """
    Jacobi two-buffer smoother for (vx, vy).
    - No allocations inside; vx_new, vy_new are required preallocated buffers (same shape as vx, vy).
    - On each iteration, compute new vx, vy from old (vx, vy), apply BCs, then swap buffers.
    - Ensures an even number of iterations so the latest solution ends up back in (vx, vy).
    - Returns references to arrays holding the latest solution (vx, vy).
    """

    # Ensure even iterations for full sweeps so results land in (vx, vy)
    max_iter += max_iter % 2

    for _ in range(max_iter):
        # vx sweep: read (vx_old_ref, vy_old_ref) -> write vx_new_ref
        _vx_jacobi_sweep(nx1, ny1, dx, dy, etap, etab,
                         vx, vy, relax_v, vx_rhs, vx_new)
        cpu_apply_vx_BC(vx_new, BC)

        # vy sweep: read (vx_old_ref, vy_old_ref) -> write vy_new_ref
        _vy_jacobi_sweep(nx1, ny1, dx, dy, etap, etab,
                         vx, vy, relax_v, vy_rhs, vy_new)
        cpu_apply_vy_BC(vy_new, BC)

        # ping-pong (reference swap; no copies)
        vx, vx_new = vx_new, vx
        vy, vy_new = vy_new, vy

    # With even iterations, latest values are in the original (vx, vy) buffers.
    # Return the references holding the latest solution.
    return vx, vy


TILE_I = 32
TILE_J = 32
T_INNER = 4

@nb.njit(inline='always')
def apply_vx_BC_ij(vx, i, j, BC):
    """
    Apply ONLY the boundary entries touched by interior (i,j) for vx.
    vx interior: i in [1..ny1-2], j in [1..nx1-3]
    """
    ny1, nx1 = vx.shape

    # Top / bottom depend on neighbor interior rows
    if i == 1:              # touching top boundary row
        vx[0, j] = -BC * vx[1, j]
    if i == ny1 - 2:        # touching bottom boundary row
        vx[ny1 - 1, j] = -BC * vx[ny1 - 2, j]

    # Left wall
    if j == 1:              # touching left boundary column
        vx[i, 0] = 0.0

    # Right + ghost columns
    if j == nx1 - 3:        # touching right boundary / ghost
        vx[i, nx1 - 2] = 0.0
        vx[i, nx1 - 1] = 0.0


@nb.njit(inline='always')
def apply_vy_BC_ij(vy, i, j, BC):
    """
    Apply ONLY the boundary entries touched by interior (i,j) for vy.
    vy interior: i in [1..ny1-3], j in [1..nx1-2]
    """
    ny1, nx1 = vy.shape

    # Left / right depend on neighbor interior columns
    if j == 1:              # touching left boundary column
        vy[i, 0] = -BC * vy[i, 1]
    if j == nx1 - 2:        # touching right boundary column
        vy[i, nx1 - 1] = -BC * vy[i, nx1 - 2]

    # Top wall
    if i == 1:              # touching top boundary row
        vy[0, j] = 0.0

    # Bottom + ghost rows
    if i == ny1 - 3:        # touching bottom boundary / ghost
        vy[ny1 - 2, j] = 0.0
        vy[ny1 - 1, j] = 0.0

@nb.njit(cache=True, inline="always")
def apply_local_vx_BC(nx1, ny1, vx, i_min, i_max, j_min, j_max, BC):
    """
    Apply boundary conditions for a local tile of vx.
    """
    
    # Apply BCs to domain [i_min, imax] x [j_min, jmax]
    if i_min == 1: # Touching top boundary row
        vx[0, j_min:j_max]  = -BC * vx[1, j_min:j_max]
    
    if i_max == ny1 - 2: # Touching bottom boundary
        vx[-1, j_min:j_max] = -BC * vx[-2, j_min:j_max]

    if j_min == 1: # Touching left boundary column
        vx[i_min:i_max, 0] = 0.0

    if j_max == nx1 - 3: # Touching right boundary
        vx[i_min:i_max, -2:] = 0.0


@nb.njit(cache=True, inline="always")
def apply_local_vy_BC(nx1, ny1, vy, i_min, i_max, j_min, j_max, BC):
    if i_min == 1: # Touching top boundary
        vy[0, j_min:j_max] = 0.0

    if i_max == ny1 - 3: # Touching bottom boundary
        vy[-1, j_min:j_max] = 0.0

    if j_min == 1: # Touching left boundary
        vy[:, 0] = -BC * vy[:, 1]

    if j_max == nx1 - 2: # Touching right boundary
        vy[:, -1] = -BC * vy[:, -2]

@nb.njit(cache=True)
def _vx_tile_kernel(ii, jj,
                    nx1, ny1,
                    dx, dy,
                    etap, etab,
                    vy, vx_rhs,
                    vx_src, vx_dst,
                    relax_v, BC):
    """
    Process one vx tile [ii:ii+TILE_I) x [jj:jj+TILE_J) with T_INNER Jacobi steps.
    Reads vy (constant for this tile pass), ping-pongs between (vx_src, vx_dst) locally.
    """
    # vx interior: i in [1..ny1-2], j in [1..nx1-3]
    i_max = min(ii + TILE_I, ny1 - 1)    # stop before ny1-1 -> last i = ny1-2
    j_max = min(jj + TILE_J, nx1 - 2)    # stop before nx1-2 -> last j = nx1-3

    # Local references we can swap without affecting caller
    src = vx_src
    dst = vx_dst

    for _ in range(T_INNER):
        for i in range(ii, i_max):
            # ensure we don't run into the guard rows
            for j in range(jj, j_max):
                etaA = etap[i,   j]
                etaB = etap[i,   j+1]
                eta1 = etab[i-1, j]
                eta2 = etab[i,   j]

                vx1, vx2, vx3, vx4, vx5, vy1, vy2, vy3, vy4 = x_momentum_coefficients(
                    dx, dy, etaA, etaB, eta1, eta2
                )

                sum_neighbors = (
                    vx1 * src[i,   j-1] +
                    vx2 * src[i-1, j  ] +
                    vx4 * src[i+1, j  ] +
                    vx5 * src[i,   j+1]
                    +
                    vy1 * vy[i-1, j  ] +
                    vy2 * vy[i,   j  ] +
                    vy3 * vy[i-1, j+1] +
                    vy4 * vy[i,   j+1]
                )

                dst[i, j] = (1.0 - relax_v) * src[i, j] + relax_v * (vx_rhs[i, j] - sum_neighbors) / vx3
                apply_vx_BC_ij(dst, i, j, BC)  # apply BCs for this tile
        # local ping-pong for this tile only
        # apply_local_vx_BC(nx1, ny1, dst, ii, i_max, jj, j_max, BC)
        src, dst = dst, src

    # No return; writes already in src/dst. For even T_INNER, results end in original vx_src.

@nb.njit(cache=True)
def _vy_tile_kernel(ii, jj,
                    nx1, ny1,
                    dx, dy,
                    etap, etab,
                    vx, vy_rhs,
                    vy_src, vy_dst,
                    relax_v, BC):
    """
    Process one vy tile [ii:ii+TILE_I) x [jj:jj+TILE_J) with T_INNER Jacobi steps.
    Reads vx (constant for this tile pass), ping-pongs between (vy_src, vy_dst) locally.
    """
    # vy interior: i in [1..ny1-3], j in [1..nx1-2]
    i_max = min(ii + TILE_I, ny1 - 2)    # stop before ny1-2 -> last i = ny1-3
    j_max = min(jj + TILE_J, nx1 - 1)    # stop before nx1-1 -> last j = nx1-2

    src = vy_src
    dst = vy_dst

    for _ in range(T_INNER):
        for i in range(ii, i_max):
            # inner loop is a good vectorization candidate for LLVM
            for j in range(jj, j_max):
            
                etaA = etap[i,   j]
                etaB = etap[i+1, j]   # y-staggered
                eta1 = etab[i,   j-1]
                eta2 = etab[i,   j]

                vy1, vy2, vy3, vy4, vy5, vx1, vx2, vx3, vx4 = y_momentum_coefficients(
                    dx, dy, etaA, etaB, eta1, eta2
                )

                sum_neighbors = (
                    vy1 * src[i,   j-1] +
                    vy2 * src[i-1, j  ] +
                    vy4 * src[i+1, j  ] +
                    vy5 * src[i,   j+1]
                    +
                    vx1 * vx[i,   j-1] +
                    vx2 * vx[i+1, j-1] +
                    vx3 * vx[i,   j  ] +
                    vx4 * vx[i+1, j  ]
                )

                dst[i, j] = (1.0 - relax_v) * src[i, j] + relax_v * (vy_rhs[i, j] - sum_neighbors) / vy3
                apply_vy_BC_ij(dst, i, j, BC)  # apply BCs for this tile
        # apply_local_vy_BC(nx1, ny1, dst, ii, i_max, jj, j_max, BC)
        src, dst = dst, src  # local ping-pong

@nb.njit(cache=True, parallel=True)
def ras_velocity_jacobi_smoother(nx1, ny1,
                                 dx, dy,
                                 etap, etab,
                                 vx, vy,
                                 relax_v, BC,
                                 vx_rhs, vy_rhs,
                                 max_iter,
                                 vx_old, vy_old,
                                 nt = nb.get_num_threads()):
    max_iter += max_iter % 2
    outer_iters = (max_iter + T_INNER) // (T_INNER + 1)

    # coverage under half-tile shifts
    # randint(0, TILE//2 + 1) -> smax = TILE//2
    smax_i = TILE_I
    smax_j = TILE_J
    Ni = (ny1 - 2)
    Nj = (nx1 - 2)
    tiles_i = (Ni + smax_i + TILE_I - 1) // TILE_I
    tiles_j = (Nj + smax_j + TILE_J - 1) // TILE_J
    total_tiles = tiles_i * tiles_j

    workers = nt if total_tiles >= nt else total_tiles

    for out in range(outer_iters):
        # phase shift
        shift_i = np.random.randint(0, smax_i + 1)
        shift_j = np.random.randint(0, smax_j + 1)
        ii0 = 1 - shift_i
        jj0 = 1 - shift_j

        # contiguous, balanced blocks (no rotation)
        base = total_tiles // workers
        rem  = total_tiles - base * workers

        for lane in nb.prange(workers):
            # first 'rem' lanes get +1 tile
            gain = 1 if lane < rem else 0
            start = lane * base + (lane if lane < rem else rem)
            count = base + gain

            # turn linear start into (ti,tj) once, then walk contiguously
            k = start
            ti = k // tiles_j
            tj = k - ti * tiles_j

            for _ in range(count):
                ii = ii0 + ti * TILE_I
                jj = jj0 + tj * TILE_J
                if ii < 1: ii = 1
                if jj < 1: jj = 1

                _vx_tile_kernel(ii, jj, nx1, ny1, dx, dy,
                                etap, etab, vy, vx_rhs,
                                vx, vx_old, relax_v, BC)

                _vy_tile_kernel(ii, jj, nx1, ny1, dx, dy,
                                etap, etab, vx_old, vy_rhs,
                                vy, vy_old, relax_v, BC)

                # advance to next tile, wrap across rows
                tj += 1
                if tj == tiles_j:
                    tj = 0
                    ti += 1
            
        # # Jacobi sweep
        # _vx_jacobi_sweep(nx1, ny1, dx, dy, etap, etab, vx_old, vy_old, relax_v, vx_rhs, vx)
        # apply_vx_BC(vx, BC)

        # # vy sweep: read (vx_old, vy_old) -> write vy_new
        # _vy_jacobi_sweep(nx1, ny1, dx, dy, etap, etab, vx_old, vy_old, relax_v, vy_rhs, vy)
        # apply_vy_BC(vy, BC)

        # vx, vx_old = vx_old, vx
        # vy, vy_old = vy_old, vy

        # Optional but often helpful for consistency:
        # apply_vx_BC(vx, BC)
        # apply_vy_BC(vy, BC)

    return vx, vy
