"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: smoother.py
Description: This file implements jacobi and red-black Gauss-Seidel smoothers
             for the Stokes flow and continuity equations in 2D.
             The pressure update is done using the Uzawa method.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numba as nb
from .utils import apply_p_BC, apply_vx_BC, apply_vy_BC

@nb.njit(cache=True, parallel=True)
def vx_jacobi_sweep(nx1, ny1,
             dx, dy,
             etap, etab,
             vx, vy, p,
             vx_new,
             relax_v, BC, rhs):
    
    # Loop only over the interior cells
    for i in nb.prange(1, ny1 - 1):
        for j in nb.prange(1, nx1 - 2):
            # Gather local (i,j) viscosity coefficients
            etaA = etap[i,   j]
            etaB = etap[i,   j+1]
            eta1 = etab[i-1, j]
            eta2 = etab[i,   j]

            # Construct coefficients for x-momentum
            vx1_coeff = 2.0 * etaA / (dx * dx)
            vx2_coeff = eta1     / (dy * dy)
            vx3_coeff = - (eta1 + eta2) / (dy * dy) - 2.0 * (etaA + etaB) / (dx * dx)
            vx4_coeff = eta2 / (dy * dy)
            vx5_coeff = 2.0 * etaB / (dx * dx)

            # Cross terms with vy
            vy1_coeff =  eta1 / (dx * dy)
            vy2_coeff = -eta2 / (dx * dy)
            vy3_coeff = -eta1 / (dx * dy)
            vy4_coeff =  eta2 / (dx * dy)

            # Pressure difference terms
            dp_right = -1.0 / dx * p[i, j+1]
            dp_left  = +1.0 / dx * p[i, j]

            
            # Sum neighbor contributions for Jacobi
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
                +
                dp_right + dp_left
            )

            diag = vx3_coeff
            
            # Jacobi update
            vx_new[i, j] = (1.0 - relax_v) * vx[i, j] + \
                           relax_v * (rhs[i, j] - sum_neighbors) / diag

    
    # Apply boundary conditions
    vx_new[0, :] = - BC * vx_new[1, :] # Top boundary
    vx_new[-1, :] = - BC * vx_new[-2, :] # Bottom boundary
    vx_new[:, 0] = 0.0   # Left boundary
    vx_new[:, -2:] = 0.0 # Right boundary + ghost cell

    # Copy solution to vx
    vx[:, :] = vx_new[:, :]

@nb.njit(cache=True, parallel=True)
def vy_jacobi_sweep(nx1, ny1,
             dx, dy,
             etap, etab,     # viscosity arrays
             vx, vy, p,      # old iteration fields
             vy_new,
             relax_v, BC, rhs):        # out array for new iteration
    
    # Loop over interior cells only
    for i in nb.prange(1, ny1 - 2):
        for j in nb.prange(1, nx1 - 1):
            # Gather local viscosities
            etaA = etap[i,   j]      # top cell's viscosity
            etaB = etap[i+1, j]      # bottom cell's viscosity
            eta1 = etab[i,   j-1]    # left edge block
            eta2 = etab[i,   j]      # right edge block

            # Construct coefficients for y-momentum
            # vy-coefficients
            vy1_coeff = eta1 / (dx * dx)
            vy2_coeff = 2.0 * etaA / (dy * dy)
            vy3_coeff = -2.0 * etaA / (dy * dy) \
                        -2.0 * etaB / (dy * dy) \
                        -eta1 / (dx * dx) \
                        -eta2 / (dx * dx)
            vy4_coeff = 2.0 * etaB / (dy * dy)
            vy5_coeff = eta2 / (dx * dx)

            # Cross terms with vx (the "off-diagonal" part)
            vx1_coeff =  eta1 / (dx * dy)
            vx2_coeff = -eta1 / (dx * dy)
            vx3_coeff = -eta2 / (dx * dy)
            vx4_coeff =  eta2 / (dx * dy)

            # Pressure gradient: - dP/dy
            dp_up   = - 1.0 / dy * p[i+1, j]   # upper node
            dp_down = + 1.0 / dy * p[i,   j]   # current node

            
            # Sum neighbor contributions
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
                +
                # pressure difference
                dp_up + dp_down
            )

            diag = vy3_coeff

            # Jacobi update for vy(i,j)
            vy_new[i, j] = (1.0 - relax_v) * vy[i, j] + \
                           relax_v * (rhs[i, j] - sum_neighbors) / diag

    
    # Apply boundary conditions
    vy_new[:, 0] = - BC * vy_new[:, 1] # Left boundary
    vy_new[:, -1] = - BC * vy_new[:, -2] # Right boundaryvy_new
    vy_new[0, :] = 0.0   # Top boundary
    vy_new[-2:, :] = 0.0  # Bottom boundary

    # Copy solution to vy    
    vy[:, :] = vy_new[:, :]

@nb.njit(cache=True, parallel=True)
def pressure_uzawa_sweep(nx1, ny1, dx, dy,
                         vx, vy, p,
                         beta,
                         relax_p, rhs,
                         p_ref = None):
    
    # 1) Update only interior cells
    for i in nb.prange(1, ny1 - 1):
        for j in nb.prange(1, nx1 - 1):
            # The continuity residual
            res = rhs[i, j] - ((vx[i, j] - vx[i, j - 1])/dx +
                               (vy[i, j] - vy[i - 1, j])/dy)

            # Point-wise update of pressure
            p[i, j] += res * beta[i, j] * relax_p

    # Anchor pressure at (1,1) if reference pressure is given
    if p_ref is not None:
        dp = p_ref - p[1, 1]
        p += dp
    
    # Apply pressure boundary conditions
    apply_p_BC(p)

@nb.njit(cache=True, parallel=True)
def pressure_uzawa_sweep_chunked(nx1, ny1, dx, dy,
                                  vx, vy, p,
                                  beta,
                                  relax_p, rhs,
                                  p_ref=None,
                                  block_size_i=16, block_size_j=16):
    """
    Chunked Uzawa pressure sweep with block-based parallelism.
    """
    n_tile_i = (ny1 - 2) // block_size_i + 1
    n_tile_j = (nx1 - 2) // block_size_j + 1

    for tile_i in nb.prange(n_tile_i):
        ii = 1 + tile_i * block_size_i
        for tile_j in range(n_tile_j):
            jj = 1 + tile_j * block_size_j
            for i in range(ii, min(ii + block_size_i, ny1 - 1)):
                for j in range(jj, min(jj + block_size_j, nx1 - 1)):
                    res = rhs[i, j] - ((vx[i, j] - vx[i, j - 1]) / dx +
                                       (vy[i, j] - vy[i - 1, j]) / dy)
                    p[i, j] += res * beta[i, j] * relax_p

    if p_ref is not None:
        dp = p_ref - p[1, 1]
        p += dp

    apply_p_BC(p)

@nb.njit(cache=True, parallel=True)
def _vx_rb_gs_sweep(nx1, ny1,
                    dx, dy,
                    etap, etab,
                    vx, vy, p,
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

            # Pressure difference terms
            dp_right = -1.0 / dx * p[i, j+1]
            dp_left  = +1.0 / dx * p[i, j]

            
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
                +
                dp_right + dp_left
            )

            diag = vx3_coeff

            # Gauss-Seidel in-place update
            vx[i, j] = (1.0 - relax_v)*vx[i, j] \
                        + relax_v*(rhs[i, j] - sum_neighbors)/diag
    # Apply vx boundary conditions
    apply_vx_BC(vx, BC)
    
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

                # Pressure difference terms
                dp_right = -1.0 / dx * p[i, j+1]
                dp_left  = +1.0 / dx * p[i, j]

                
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
                    +
                    dp_right + dp_left
                )

                diag = vx3_coeff

                # Gauss-Seidel in-place update
                vx[i, j] = (1.0 - relax_v)*vx[i, j] \
                           + relax_v*(rhs[i, j] - sum_neighbors)/diag
    
    # Apply vx boundary conditions
    apply_vx_BC(vx, BC)

@nb.njit(cache=True, parallel=True)
def _vy_red_black_gs_sweep(nx1, ny1,
                           dx, dy,
                           etap, etab,
                           vx, vy, p,
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

                # Pressure gradient: - dP/dy
                dp_up   = -1.0/dy * p[i+1, j]
                dp_down = +1.0/dy * p[i,   j]

                
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
                    +
                    # pressure difference
                    dp_up + dp_down
                )

                diag = vy3_coeff
            
                # 4) Gauss-Seidel in-place update
                vy[i, j] = (1.0 - relax_v)*vy[i, j] \
                           + relax_v*(rhs[i, j] - sum_neighbors)/diag
    # Apply vy boundary conditions
    apply_vy_BC(vy, BC)

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

                # Pressure gradient
                dp_up   = -1.0/dy * p[i+1, j]
                dp_down = +1.0/dy * p[i,   j]

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
                    +
                    dp_up + dp_down
                )

                diag = vy3_coeff

                # 4) In-place update                
                vy[i, j] = (1.0 - relax_v)*vy[i, j] \
                           + relax_v*(rhs[i, j] - sum_neighbors)/diag
    
    # Apply vy boundary conditions
    apply_vy_BC(vy, BC)

@nb.njit(cache=True)
def red_black_gs(nx1, ny1,
                 dx, dy,
                 etap, etab,
                 vx, vy, p,
                 relax_v, relax_p,
                 p_ref, BC, p_rhs, vx_rhs, vy_rhs, max_iter):
    """
    Full red-black Gauss-Seidel smoother for the velocity fields.
    """
    for _ in range(max_iter):
        # Solve velocity system approximately
        for __ in range(3):
            _vx_rb_gs_sweep(nx1, ny1,
                            dx, dy,
                            etap, etab,
                            vx, vy, p,
                            relax_v, vx_rhs, BC)
            
            _vy_red_black_gs_sweep(nx1, ny1,
                                dx, dy,
                                etap, etab,
                                vx, vy, p,
                                relax_v, vy_rhs, BC)
        
        
        # Apply Uzawa pressure update
        pressure_uzawa_sweep(nx1, ny1, dx, dy,
                            vx, vy, p, etap,
                            relax_p, p_rhs, p_ref)
        
@nb.njit(cache=True, parallel=True)
def _vx_rb_gs_sweep_chunked(nx1, ny1,
                            dx, dy,
                            etap, etab,
                            vx, vy, p,
                            relax_v, rhs, BC,
                            block_size_i=64, block_size_j=64):
    """
    Red-Black Gauss-Seidel update for vx with chunked blocks for better parallel scalability.
    """
    # Red pass
    n_tile_i = (ny1 - 2) // block_size_i + 1
    for tile_i in nb.prange(n_tile_i):
        ii = 1 + tile_i * block_size_i
        for jj in range(1, nx1 - 2, block_size_j):
            for i in range(ii, min(ii + block_size_i, ny1 - 1)):
                j_start = 1 if i % 2 == 0 else 2
                for j in range(jj + (j_start - jj) % 2, min(jj + block_size_j, nx1 - 2), 2):
                    etaA = etap[i, j]
                    etaB = etap[i, j+1]
                    eta1 = etab[i-1, j]
                    eta2 = etab[i, j]

                    vx1_coeff = 2.0 * etaA / (dx * dx)
                    vx2_coeff = eta1 / (dy * dy)
                    vx3_coeff = -(eta1 + eta2) / (dy * dy) - 2.0 * (etaA + etaB) / (dx * dx)
                    vx4_coeff = eta2 / (dy * dy)
                    vx5_coeff = 2.0 * etaB / (dx * dx)

                    vy1_coeff = eta1 / (dx * dy)
                    vy2_coeff = -eta2 / (dx * dy)
                    vy3_coeff = -eta1 / (dx * dy)
                    vy4_coeff = eta2 / (dx * dy)

                    dp_right = -1.0 / dx * p[i, j+1]
                    dp_left = +1.0 / dx * p[i, j]

                    sum_neighbors = (
                        vx1_coeff * vx[i, j-1] +
                        vx2_coeff * vx[i-1, j] +
                        vx4_coeff * vx[i+1, j] +
                        vx5_coeff * vx[i, j+1] +
                        vy1_coeff * vy[i-1, j] +
                        vy2_coeff * vy[i, j] +
                        vy3_coeff * vy[i-1, j+1] +
                        vy4_coeff * vy[i, j+1] +
                        dp_right + dp_left
                    )

                    vx[i, j] = (1.0 - relax_v) * vx[i, j] + relax_v * (rhs[i, j] - sum_neighbors) / vx3_coeff

    apply_vx_BC(vx, BC)

    # Black pass
    n_tile_i = (ny1 - 2) // block_size_i + 1
    for tile_i in nb.prange(n_tile_i):
        ii = 1 + tile_i * block_size_i
        for jj in range(1, nx1 - 2, block_size_j):
            for i in range(ii, min(ii + block_size_i, ny1 - 1)):
                j_start = 2 if i % 2 == 0 else 1
                for j in range(jj + (j_start - jj) % 2, min(jj + block_size_j, nx1 - 2), 2):
                    etaA = etap[i, j]
                    etaB = etap[i, j+1]
                    eta1 = etab[i-1, j]
                    eta2 = etab[i, j]

                    vx1_coeff = 2.0 * etaA / (dx * dx)
                    vx2_coeff = eta1 / (dy * dy)
                    vx3_coeff = -(eta1 + eta2) / (dy * dy) - 2.0 * (etaA + etaB) / (dx * dx)
                    vx4_coeff = eta2 / (dy * dy)
                    vx5_coeff = 2.0 * etaB / (dx * dx)

                    vy1_coeff = eta1 / (dx * dy)
                    vy2_coeff = -eta2 / (dx * dy)
                    vy3_coeff = -eta1 / (dx * dy)
                    vy4_coeff = eta2 / (dx * dy)

                    dp_right = -1.0 / dx * p[i, j+1]
                    dp_left = +1.0 / dx * p[i, j]

                    sum_neighbors = (
                        vx1_coeff * vx[i, j-1] +
                        vx2_coeff * vx[i-1, j] +
                        vx4_coeff * vx[i+1, j] +
                        vx5_coeff * vx[i, j+1] +
                        vy1_coeff * vy[i-1, j] +
                        vy2_coeff * vy[i, j] +
                        vy3_coeff * vy[i-1, j+1] +
                        vy4_coeff * vy[i, j+1] +
                        dp_right + dp_left
                    )

                    vx[i, j] = (1.0 - relax_v) * vx[i, j] + relax_v * (rhs[i, j] - sum_neighbors) / vx3_coeff

    apply_vx_BC(vx, BC)


@nb.njit(cache=True, parallel=True)
def _vy_red_black_gs_sweep_chunked(nx1, ny1,
                                   dx, dy,
                                   etap, etab,
                                   vx, vy, p,
                                   relax_v, rhs, BC,
                                   block_size_i=64, block_size_j=64):
    """
    Chunked Red-Black Gauss-Seidel update for vy.
    """
    # Red pass
    n_tile_i = (ny1 - 3) // block_size_i + 1  # because range goes to ny1 - 2
    for tile_i in nb.prange(n_tile_i):  # valid prange loop
        ii = 1 + tile_i * block_size_i
        for jj in range(1, nx1 - 1, block_size_j):
            for i in range(ii, min(ii + block_size_i, ny1 - 2)):
                j_start = 1 if i % 2 == 0 else 2
                for j in range(jj + (j_start - jj) % 2, min(jj + block_size_j, nx1 - 1), 2):
                    etaA = etap[i, j]
                    etaB = etap[i+1, j]
                    eta1 = etab[i, j-1]
                    eta2 = etab[i, j]

                    vy1_coeff = eta1 / (dx * dx)
                    vy2_coeff = 2.0 * etaA / (dy * dy)
                    vy3_coeff = -2.0 * etaA/(dy*dy) - 2.0 * etaB/(dy*dy) - eta1/(dx*dx) - eta2/(dx*dx)
                    vy4_coeff = 2.0 * etaB / (dy * dy)
                    vy5_coeff = eta2 / (dx * dx)

                    vx1_coeff = eta1 / (dx * dy)
                    vx2_coeff = -eta1 / (dx * dy)
                    vx3_coeff = -eta2 / (dx * dy)
                    vx4_coeff = eta2 / (dx * dy)

                    dp_up = -1.0/dy * p[i+1, j]
                    dp_down = +1.0/dy * p[i, j]

                    sum_neighbors = (
                        vy1_coeff * vy[i, j-1] +
                        vy2_coeff * vy[i-1, j] +
                        vy4_coeff * vy[i+1, j] +
                        vy5_coeff * vy[i, j+1] +
                        vx1_coeff * vx[i, j-1] +
                        vx2_coeff * vx[i+1, j-1] +
                        vx3_coeff * vx[i, j] +
                        vx4_coeff * vx[i+1, j] +
                        dp_up + dp_down
                    )

                    vy[i, j] = (1.0 - relax_v) * vy[i, j] + relax_v * (rhs[i, j] - sum_neighbors) / vy3_coeff

    apply_vy_BC(vy, BC)

    # Black pass
    n_tile_i = (ny1 - 3) // block_size_i + 1  # because range goes to ny1 - 2
    for tile_i in nb.prange(n_tile_i):  # valid prange loop
        ii = 1 + tile_i * block_size_i
        for jj in range(1, nx1 - 1, block_size_j):
            for i in range(ii, min(ii + block_size_i, ny1 - 2)):
                j_start = 2 if i % 2 == 0 else 1
                for j in range(jj + (j_start - jj) % 2, min(jj + block_size_j, nx1 - 1), 2):
                    etaA = etap[i, j]
                    etaB = etap[i+1, j]
                    eta1 = etab[i, j-1]
                    eta2 = etab[i, j]

                    vy1_coeff = eta1 / (dx * dx)
                    vy2_coeff = 2.0 * etaA / (dy * dy)
                    vy3_coeff = -2.0 * etaA/(dy*dy) - 2.0 * etaB/(dy*dy) - eta1/(dx*dx) - eta2/(dx*dx)
                    vy4_coeff = 2.0 * etaB / (dy * dy)
                    vy5_coeff = eta2 / (dx * dx)

                    vx1_coeff = eta1 / (dx * dy)
                    vx2_coeff = -eta1 / (dx * dy)
                    vx3_coeff = -eta2 / (dx * dy)
                    vx4_coeff = eta2 / (dx * dy)

                    dp_up = -1.0/dy * p[i+1, j]
                    dp_down = +1.0/dy * p[i, j]

                    sum_neighbors = (
                        vy1_coeff * vy[i, j-1] +
                        vy2_coeff * vy[i-1, j] +
                        vy4_coeff * vy[i+1, j] +
                        vy5_coeff * vy[i, j+1] +
                        vx1_coeff * vx[i, j-1] +
                        vx2_coeff * vx[i+1, j-1] +
                        vx3_coeff * vx[i, j] +
                        vx4_coeff * vx[i+1, j] +
                        dp_up + dp_down
                    )

                    vy[i, j] = (1.0 - relax_v) * vy[i, j] + relax_v * (rhs[i, j] - sum_neighbors) / vy3_coeff

    apply_vy_BC(vy, BC)


def red_black_gs_chunked(nx1, ny1,
                 dx, dy,
                 etap, etab,
                 vx, vy, p,
                 relax_v, relax_p,
                 p_ref, BC, p_rhs, vx_rhs, vy_rhs, max_iter, chunk_size=32):
    """
    Full red-black Gauss-Seidel smoother for the velocity fields.
    """
    for _ in range(max_iter):
        # Solve velocity system approximately
        for __ in range(3):
            _vx_rb_gs_sweep_chunked(nx1, ny1,
                            dx, dy,
                            etap, etab,
                            vx, vy, p,
                            relax_v, vx_rhs, BC, block_size_i=chunk_size, block_size_j=chunk_size)
            
            _vy_red_black_gs_sweep_chunked(nx1, ny1,
                                dx, dy,
                                etap, etab,
                                vx, vy, p,
                                relax_v, vy_rhs, BC, block_size_i=chunk_size, block_size_j=chunk_size)
        
        # Apply Uzawa pressure update
        pressure_uzawa_sweep_chunked(nx1, ny1, dx, dy,
                            vx, vy, p, etap,
                            relax_p, p_rhs, p_ref, chunk_size)