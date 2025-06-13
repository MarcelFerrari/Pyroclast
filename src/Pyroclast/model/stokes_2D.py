"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: stokes+continuity_markers.py
Description: This file implements basic Stokes flow and continuity equations
             for boyancy-driven flow with marker-in-cell method.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
import numba as nb
import numpy as np


from Pyroclast.model.base_model import BaseModel
from Pyroclast.profiling import timer
import scipy.sparse as sp
from logging import get_logger

logger = get_logger(__name__)

# Model class
class IncompressibleStokes2D(BaseModel):
    """
    Basic Stokes flow and continuity equations for buoyancy-driven flow in 2D.
    
    This class solves the Stokes and continuity equations for a fluid on a 2D
    domain.

    The domain of choice is a rectangular box with a circular inclusion of
    different viscosity and density. 

    We use a basic 2D staggered grid with uniform spacing in each dimension.
    """

    def __init__(self, ctx):
        # At this point the grid and markers are already initialized
        s, p, o = ctx

        # Extra options (with default values if not set in the input file)
        self.BC = p.get('BC', -1)      # Boundary condition parameter
        self.gy = p.get('gy', 10.0)    # Gravity constant

        # Set up the matrix storage for the Stokes equations
        self.n_rows = s.nx1*s.ny1*3

        # Set up rho, eta_b, and eta_p arrays
        # These are not defined on ghost nodes
        s.rho = np.zeros((s.ny1, s.nx1))
        s.etab = np.zeros((s.ny1, s.nx1))
        s.etap = np.zeros((s.ny1, s.nx1))

        # Set nan on borders
        s.rho[-1, :] = np.nan
        s.rho[:, -1] = np.nan
        s.etab[-1, :] = np.nan
        s.etab[:, -1] = np.nan
        s.etap[-1, :] = np.nan
        s.etap[:, -1] = np.nan
        
        # Set up variables for solution quantities
        # These are defined on ghost nodes to enforce the boundary conditions
        s.p = np.zeros((s.ny1, s.nx1))
        s.vx = np.zeros((s.ny1, s.nx1))
        s.vy = np.zeros((s.ny1, s.nx1))

        self.frame = 0
        self.zpad = len(str(p.max_iterations//o.framedump_interval)) + 1

    def update_time_step(self, ctx):
        # Read the context
        s, p, o = ctx

        # Get maximum velocity
        vxmax = np.max(np.abs(s.vx))
        vymax = np.max(np.abs(s.vy))
        dty = p.cfl_dispmax * p.L / vymax
        dtx = p.cfl_dispmax * p.L / vxmax
        
        # It is important to set dt in the context
        # parameters so that it can be accessed by other
        # classes in the simulation.
        s.dt = min(dtx, dty)

    def scale_and_solve(self, Acsc, rhs):
        # Compute row norms (for D_r) using CSR
        Acsr = Acsc.tocsr()
        Dr = row_norms_csr(Acsr.data, Acsr.indptr, self.n_rows)
        
        # Compute column norms (for D_c) using CSC
        Dc = col_norms_csc(Acsc.data, Acsc.indptr, self.n_rows)

        # Avoid division by zero
        Dr[Dr == 0] = 1.0
        Dc[Dc == 0] = 1.0

        # Build diagonal scaling matrices
        D_r_inv = sp.diags(1.0 / Dr)  # shape (n_rows, n_rows)
        D_c_inv = sp.diags(1.0 / Dc)  # shape (n_cols, n_cols)

        # Scale the matrix: A_scaled = D_r^{-1} * A * D_c^{-1}
        A_scaled = D_r_inv @ Acsc @ D_c_inv

        # Scale the RHS: rhs_scaled = D_r^{-1} * rhs
        rhs_scaled = (1.0 / Dr) * rhs

        # Solve the scaled system
        x_scaled = sp.linalg.spsolve(A_scaled, rhs_scaled)

        # Rescale the solution: x = D_c^{-1} * x_scaled
        x = (1.0 / Dc) * x_scaled

        return x
     
    def solve(self, ctx):
        # Read the context
        s, p, o = ctx

        # Assemble the matrix
        i_idx, j_idx, vals, rhs = assemble(s.nx1, s.ny1,
                                           s.dx, s.dy,
                                           p.gy,
                                           s.etap, s.etab, s.rho,
                                           p.p_ref,
                                           self.BC)
        
        # Assemble the matrix in COO format
        A = sp.coo_matrix((vals, (i_idx, j_idx)), shape=(self.n_rows, self.n_rows))

        # Convert to CSC format for efficient solving
        A = A.tocsc()

        # Call spsolve with explicit types
        with timer.time_section("Model Solve", "Stokes Solve"):
            # Solve the system of equations
            u = self.scale_and_solve(A, rhs)

        # Compute the residual
        res = np.linalg.norm(rhs - A @ u)/np.linalg.norm(rhs)
        logger.debug(f"Residual: {res}")

        # Extract solution quantities
        u = u.reshape((s.ny1, s.nx1, 3))
        
        # In-place copy of solution quantities
        s.p[...] = u[:, :, 0]
        s.vx[...] = u[:, :, 1]
        s.vy[...] = u[:, :, 2]
        
    def dump(self, ctx):
        s, p, o = ctx

        # Dump state to file
        with open(f"frame_{str(self.frame).zfill(self.zpad)}.npz", 'wb') as f:
            np.savez(f, vx=s.vx, vy=s.vy, p=s.p,
                    rho=s.rho, etab=s.etab, etap=s.etap)
        
        logger.info(f"Frame {self.frame} written to file.")
        self.frame += 1 # Increment frame counter


    def finalize(self, ctx):
        # Read the context
        s, p, o = ctx

        # Write rho, eta_b, eta_p, vx, vy and p to npz file
        logger.info("Writing solution to file...")
        np.savez("solution.npz", p=s.p, vx=s.vx, vy=s.vy, rho=s.rho, etab=s.etab, etap=s.etap)
                
# Numba compiled functions - Not part of the class
@nb.njit(cache = True)
def idx(nx1, i, j, q):
    # Helper function to map 2D indices to 1D index
    # i: matrix row index (y-index)
    # j: matrix column index (x-index)
    # q: variable index (0: P, 1: vx, 2: vy)
    return 3*(i*nx1 + j) + q
    
@nb.njit(cache = True)
def insert(mat, i, j, v):
    # Mat is a tuple (i_idx, j_idx, vals)
    cur = mat[3][0]
    mat[0][cur] = i
    mat[1][cur] = j
    mat[2][cur] = v

    # Increment current index
    mat[3][0] += 1
    
@timer.time_function("Model Solve", "Assemble")
@nb.njit(cache=True)
def assemble(nx1, ny1, dx, dy, gy, etap, etab, rho, p_ref, BC):
    # Assemble matrix in COO format
    n_eqs = 3               # Number of equations to solve
    n_rows = nx1*ny1*n_eqs  # Number of rows in the matrix        
    max_nnz = 12            # Maximum number of non-zero elements (~12 per row)

    # Preallocate memory for COO format
    i_idx = np.zeros((max_nnz*n_rows,), dtype=np.int32)
    j_idx = np.zeros((max_nnz*n_rows,), dtype=np.int32)
    vals = np.zeros((max_nnz*n_rows,), dtype=np.float64)
    mat = (i_idx, j_idx, vals, np.array([0], dtype=np.int32))
    b = np.zeros((n_rows,), dtype=np.float64)

    # Loop over the grid
    for i in range(ny1):
        for j in range(nx1):
            # Continuity equation (P)
            kij = idx(nx1, i, j, 0)

            # Set P = 0 for ghost nodes
            if i == 0 or j == 0 or i == ny1 - 1 or j == nx1 - 1:
                insert(mat, kij, kij, 1.0)
                b[kij] = 0.0
            elif i == 1 and j == 1:
                insert(mat, kij, kij, 1.0)
                b[kij] = p_ref
            else:
                # vx coefficients
                insert(mat, kij, idx(nx1, i, j, 1), 1.0/dx)
                insert(mat, kij, idx(nx1, i, j-1, 1), -1.0/dx)

                # vy coefficients
                insert(mat, kij, idx(nx1, i, j, 2), 1.0/dy)
                insert(mat, kij, idx(nx1, i-1, j, 2), -1.0/dy)

                # RHS
                b[kij] = 0.0

            # 2) x-momentum equation (vx)
            kij = idx(nx1, i, j, 1)
            
            if j == 0 or j >= nx1-2: # Last two nodes in x-direction
                insert(mat, kij, kij, 1.0)
                b[kij] = 0.0
            elif i == 0 or i == ny1 - 1: # First and last nodes in y-direction
                insert(mat, kij, idx(nx1, i, j, 1), 1.0)
                if i == 0: # Top boundary
                    insert(mat, kij, idx(nx1, i+1, j, 1), BC)
                else: # Bottom boundary
                    insert(mat, kij, idx(nx1, i-1, j, 1), BC)
                b[kij] = 0.0
            else:
                # Extract viscosity values
                etaA = etap[i, j]
                etaB = etap[i, j+1]
                eta1 = etab[i-1, j]
                eta2 = etab[i, j]

                # vx coefficients
                vx1_coeff = 2*etaA/dx**2
                vx2_coeff = eta1/dy**2
                vx3_coeff = -eta1/dy**2 - eta2/dy**2 - 2*etaA/dx**2 - 2*etaB/dx**2
                vx4_coeff = eta2/dy**2
                vx5_coeff = 2*etaB/dx**2

                # Store vx coefficients
                insert(mat, kij, idx(nx1, i, j-1, 1), vx1_coeff) #vx1 = vx(i, j-1)
                insert(mat, kij, idx(nx1, i-1, j, 1), vx2_coeff) #vx2 = vx(i-1, j)
                insert(mat, kij, idx(nx1, i, j, 1), vx3_coeff) #vx3 = vx(i, j)
                insert(mat, kij, idx(nx1, i+1, j, 1), vx4_coeff) #vx4 = vx(i+1, j)
                insert(mat, kij, idx(nx1, i, j+1, 1), vx5_coeff) #vx5 = vx(i, j+1)

                # vy coefficients
                vy1_coeff = eta1/(dx*dy)
                vy2_coeff = -eta2/(dx*dy)
                vy3_coeff = -eta1/(dx*dy)
                vy4_coeff = eta2/(dx*dy)

                #Store vy coefficients
                insert(mat, kij, idx(nx1, i-1, j, 2), vy1_coeff)   #vy1 = vy(i-1, j)
                insert(mat, kij, idx(nx1, i, j, 2), vy2_coeff)     #vy2 = vy(i, j)
                insert(mat, kij, idx(nx1, i-1, j+1, 2), vy3_coeff) #vy3 = vy(i-1, j+1)
                insert(mat, kij, idx(nx1, i, j+1, 2), vy4_coeff)   #vy4 = vy(i, j+1)
                        
                # -dP/dx
                insert(mat, kij, idx(nx1, i, j+1, 0), -1.0/dx)
                insert(mat, kij, idx(nx1, i, j, 0), 1.0/dx)
                
                # RHS
                b[kij] = 0.0

            # 3) y-momentum equation (vy)
            kij = idx(nx1, i, j, 2)

            if i == 0 or i >= ny1 - 2: # Last two nodes in y-direction
                insert(mat, kij, kij, 1.0)
                b[kij] = 0.0
            elif j == 0 or j == nx1 - 1:
                insert(mat, kij, kij, 1.0)
                if j == 0:
                    insert(mat, kij, idx(nx1, i, j+1, 2), BC)
                else:
                    insert(mat, kij, idx(nx1, i, j-1, 2), BC)
                b[kij] = 0.0
            else:
                # Extract viscosity values
                etaA = etap[i, j]
                etaB = etap[i+1, j]
                eta1 = etab[i, j-1]
                eta2 = etab[i, j]

                # vy coefficients
                vy1_coeff = eta1/dx**2
                vy2_coeff = 2*etaA/dy**2
                vy3_coeff = -2*etaA/dy**2 - 2*etaB/dy**2 - eta1/dx**2 - eta2/dx**2
                vy4_coeff = 2*etaB/dy**2
                vy5_coeff = eta2/dx**2

                # Store vy coefficients
                insert(mat, kij, idx(nx1, i, j-1, 2), vy1_coeff) #vy1 = vy(i, j-1)
                insert(mat, kij, idx(nx1, i-1, j, 2), vy2_coeff) #vy2 = vy(i-1, j)
                insert(mat, kij, idx(nx1, i, j, 2), vy3_coeff) #vy3 = vy(i, j)
                insert(mat, kij, idx(nx1, i+1, j, 2), vy4_coeff) #vy4 = vy(i+1, j)
                insert(mat, kij, idx(nx1, i, j+1, 2), vy5_coeff) #vy5 = vy(i, j+1)

                #vx coefficients
                vx1_coeff = eta1/(dx*dy)
                vx2_coeff = -eta1/(dx*dy)
                vx3_coeff = -eta2/(dx*dy)
                vx4_coeff = eta2/(dx*dy)

                # Store vx coefficients
                insert(mat, kij, idx(nx1, i, j-1, 1), vx1_coeff) #vx1 = vx(i, j-1)
                insert(mat, kij, idx(nx1, i+1, j-1, 1), vx2_coeff) #vx2 = vx(i+1, j-1)
                insert(mat, kij, idx(nx1, i, j, 1), vx3_coeff) #vx3 = vx(i, j)
                insert(mat, kij, idx(nx1, i+1, j, 1), vx4_coeff) #vx4 = vx(i+1, j)

                # -dP/dy
                insert(mat, kij, idx(nx1, i+1, j, 0), -1.0/dy)
                insert(mat, kij, idx(nx1, i, j, 0), 1.0/dy)
                
                # RHS
                b[kij] = -gy*rho[i, j]
    return mat[0], mat[1], mat[2], b

# Matrix scaling functions
@nb.njit(cache=True)
def row_norms_csr(data, indptr, n_rows):
    norms = np.empty(n_rows)
    for i in range(n_rows):
        start = indptr[i]
        end = indptr[i+1]
        s = 0.0
        for k in range(start, end):
            s += data[k] * data[k]
        norms[i] = np.sqrt(s)
    return norms

@nb.njit(cache=True)
def col_norms_csc(data, indptr, n_cols):
    norms = np.empty(n_cols)
    for j in range(n_cols):
        start = indptr[j]
        end = indptr[j+1]
        s = 0.0
        for k in range(start, end):
            s += data[k] * data[k]
        norms[j] = np.sqrt(s)
    return norms