"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: stokes+continuity_markers.py
Description: This file implements basic Stokes flow and continuity equations
             for boyancy-driven flow with marker-in-cell method.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
import numba as nb
import numpy as np

from Pyroclast.linalg import xp
from Pyroclast.model.base_model import BaseModel
from Pyroclast.interpolation.linear_2D_cpu \
    import interpolate_markers2grid as interpolate
from Pyroclast.profiling import timer

# Initialize solver
import phasma as ph
solver = ph.direct.Eigen_SparseLU(ph.Scale.Full)

# Model class
class StokesContinuity2D(BaseModel): # Inherit from BaseModel
                                     # this automatically does some magic
    """
    Basic Stokes flow and continuity equations for buoyancy-driven flow in 2D.
    
    This class solves the Stokes and continuity equations for a fluid on a 2D
    domain.

    The domain of choice is a rectangular box with a circular inclusion of
    different viscosity and density. 

    We use a basic 2D staggered grid with uniform spacing in each dimension.
    """

    def initialize(self):
        # Initialize the quantities of interest
        # At this point the grid and markers are already initialized

        # Store parameters for convenience
        # The variables defined as attributes of "self" in the grid
        # and pool objects can be accessed here as well via the context "ctx".

        # Time integration properties
        self.dispmax = self.ctx.params.get('dispmax', 0.5)

        # Grid properties
        self.xsize = self.ctx.params.xsize
        self.ysize = self.ctx.params.ysize
        self.nx = self.ctx.params.nx
        self.ny = self.ctx.params.ny
        self.nx1 = self.nx + 1
        self.ny1 = self.ny + 1
        self.dx = self.ctx.grid.dx
        self.dy = self.ctx.grid.dy
        
        # Frame counter and dump frequency
        self.dump_interval = self.ctx.params.get('dump_interval', 1)
        self.frame = 0

        # Marker properties
        self.nm = self.ctx.pool.nm

        # Extra options (with default values if not set in the input file)
        self.BC = self.ctx.params.get('BC', -1)      # Boundary condition parameter
        self.gy = self.ctx.params.get('gy', 10.0)    # Gravity constant

        # Set up the matrix storage for the Stokes equations
        self.n_rows = self.nx1*self.ny1*3

        # Set up rho, eta_b, and eta_p arrays
        self.rho = xp.zeros((self.ny, self.nx))
        self.etab = xp.zeros((self.ny, self.nx))
        self.etap = xp.zeros((self.ny, self.nx))
        self.p_ref = self.ctx.params.p_ref
        self.p_scale = self.ctx.params.p_scale
        self.v_scale = self.ctx.params.v_scale

        # Set up variables for solution quantities (this is just for reference)
        self.p = None
        self.vx = None
        self.vy = None

    def update_time_step(self):
        # Get maximum velocity
        vxmax = xp.max(xp.abs(self.vx))
        vymax = xp.max(xp.abs(self.vy))
        dty = self.dispmax * self.ctx.grid.dy / vymax
        dtx = self.dispmax * self.ctx.grid.dx / vxmax
        
        # It is important to set dt in the context
        # parameters so that it can be accessed by other
        # classes in the simulation.
        self.ctx.params.dt = min(dtx, dty)

    @timer.time_function("Model Solve", "Interpolation")
    def interpolate(self):
        """
        Interpolate density and viscosity from markers to grid nodes.
        """

        self.rho = interpolate(self.ctx.grid.xvy,           # Density on y-velocity nodes
                                self.ctx.grid.yvy, 
                                self.ctx.pool.xm,           # Marker x positions
                                self.ctx.pool.ym,           # Marker y positions
                                (self.ctx.pool.rhom,),      # Marker density
                                indexing="equidistant",     # Equidistant grid spacing
                                return_weights=False)       # Do not return weights
        
        self.etab = interpolate(self.ctx.grid.x,            # Basic viscosity on grid nodes
                                 self.ctx.grid.y,
                                 self.ctx.pool.xm,          # Marker x positions
                                 self.ctx.pool.ym,          # Marker y positions
                                 (self.ctx.pool.etam,),     # Marker viscosity
                                 indexing="equidistant",    # Equidistant grid spacing
                                 return_weights=False)      # Do not return weights
        
        self.etap = interpolate(self.ctx.grid.xp,      # Pressure viscosity on grid nodes
                                 self.ctx.grid.yp,
                                 self.ctx.pool.xm,          # Marker x positions
                                 self.ctx.pool.ym,          # Marker y positions
                                 (self.ctx.pool.etam,),     # Marker viscosity
                                 indexing="equidistant",    # Equidistant grid spacing
                                 return_weights=False)      # Do not return weights
     
    def solve(self):
        # Assemble the matrix
        i_idx, j_idx, vals, rhs = assemble(self.nx1, self.ny1, self.dx, self.dy,
                                self.gy, self.etap, self.etab, self.rho,
                                self.p_ref, self.p_scale, self.v_scale,
                                self.BC)
        
        A = ph.CCSMatrix(i_idx, j_idx, vals)
        
        # Call spsolve with explicit types
        with timer.time_section("Model Solve", "spSolve"):
            solver.factorize(A)
            u = solver.solve(rhs)

        # Compute the residual
        res = np.linalg.norm(rhs - A * u)/np.linalg.norm(rhs)
        print("Residual: ", res)

        # Extract solution quantities
        u = u.reshape((self.ny1, self.nx1, 3))
        self.p = u[:, :, 0]
        self.vx = u[:, :, 1]
        self.vy = u[:, :, 2]

        # Dump p, vx, vy to npz file
        if self.frame % self.dump_interval == 0:
            with open(f"frame_{str(self.frame).zfill(4)}.npz", 'wb') as f:
                xp.savez(f, p=self.p, vx=self.vx, vy=self.vy, rho=self.rho, etab=self.etab, etap=self.etap)
        self.frame += 1
        
    def finalize(self):
        # Write rho, eta_b, eta_p, vx, vy and p to npz file
        print("Writing solution to file...")
        np.savez("solution.npz", rho=self.rho, etab=self.etab, etap=self.etap,
                 vx=self.vx, vy=self.vy, p=self.p)
                
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
def assemble(nx1, ny1, dx, dy, gy, etap, etab, rho,
             p_ref, p_scale, v_scale,
             BC):
    # Assemble matrix in COO format
    n_eqs = 3               # Number of equations to solve
    n_rows = nx1*ny1*n_eqs  # Number of rows in the matrix        
    max_nnz = 12            # Maximum number of non-zero elements (~12 per row)

    # Preallocate memory for COO format
    i_idx = np.zeros((max_nnz*n_rows,), dtype=xp.int32)
    j_idx = np.zeros((max_nnz*n_rows,), dtype=xp.int32)
    vals = np.zeros((max_nnz*n_rows,), dtype=xp.float64)
    mat = (i_idx, j_idx, vals, np.array([0], dtype=xp.int32))
    b = np.zeros((n_rows,), dtype=xp.float64)

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
                insert(mat, kij, kij, p_scale)
                b[kij] = p_ref
            else:
                # vx coefficients
                insert(mat, kij, idx(nx1, i, j, 1), v_scale/dx)
                insert(mat, kij, idx(nx1, i, j-1, 1), -v_scale/dx)

                # vy coefficients
                insert(mat, kij, idx(nx1, i, j, 2), v_scale/dy)
                insert(mat, kij, idx(nx1, i-1, j, 2), -v_scale/dy)

                # RHS
                b[kij] = 0.0

            # 2) x-momentum equation (vx)
            kij = idx(nx1, i, j, 1)
            
            if j == 0 or j >= nx1-2: # Last two nodes in x-direction
                insert(mat, kij, kij, 1.0)
                b[kij] = 0.0
            elif i == 0 or i == ny1 - 1: # First and last nodes in y-direction
                insert(mat, kij, idx(nx1, i, j, 1), v_scale)
                if i == 0: # Top boundary
                    insert(mat, kij, idx(nx1, i+1, j, 1), BC*v_scale)
                else: # Bottom boundary
                    insert(mat, kij, idx(nx1, i-1, j, 1), BC*v_scale)
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
                insert(mat, kij, idx(nx1, i, j-1, 1), vx1_coeff*v_scale) #vx1 = vx(i, j-1)
                insert(mat, kij, idx(nx1, i-1, j, 1), vx2_coeff*v_scale) #vx2 = vx(i-1, j)
                insert(mat, kij, idx(nx1, i, j, 1), vx3_coeff*v_scale) #vx3 = vx(i, j)
                insert(mat, kij, idx(nx1, i+1, j, 1), vx4_coeff*v_scale) #vx4 = vx(i+1, j)
                insert(mat, kij, idx(nx1, i, j+1, 1), vx5_coeff*v_scale) #vx5 = vx(i, j+1)

                # vy coefficients
                vy1_coeff = eta1/(dx*dy)
                vy2_coeff = -eta2/(dx*dy)
                vy3_coeff = -eta1/(dx*dy)
                vy4_coeff = eta2/(dx*dy)

                #Store vy coefficients
                insert(mat, kij, idx(nx1, i-1, j, 2), vy1_coeff*v_scale)   #vy1 = vy(i-1, j)
                insert(mat, kij, idx(nx1, i, j, 2), vy2_coeff*v_scale)     #vy2 = vy(i, j)
                insert(mat, kij, idx(nx1, i-1, j+1, 2), vy3_coeff*v_scale) #vy3 = vy(i-1, j+1)
                insert(mat, kij, idx(nx1, i, j+1, 2), vy4_coeff*v_scale)   #vy4 = vy(i, j+1)
                        
                # -dP/dx
                insert(mat, kij, idx(nx1, i, j+1, 0), -p_scale/dx)
                insert(mat, kij, idx(nx1, i, j, 0), p_scale/dx)
                
                # RHS
                b[kij] = 0.0

            # 3) y-momentum equation (vy)
            kij = idx(nx1, i, j, 2)

            if i == 0 or i >= ny1 - 2: # Last two nodes in y-direction
                insert(mat, kij, kij, 1.0)
                b[kij] = 0.0
            elif j == 0 or j == nx1 - 1:
                insert(mat, kij, kij, v_scale)
                if j == 0:
                    insert(mat, kij, idx(nx1, i, j+1, 2), BC*v_scale)
                else:
                    insert(mat, kij, idx(nx1, i, j-1, 2), BC*v_scale)
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
                insert(mat, kij, idx(nx1, i, j-1, 2), vy1_coeff*v_scale) #vy1 = vy(i, j-1)
                insert(mat, kij, idx(nx1, i-1, j, 2), vy2_coeff*v_scale) #vy2 = vy(i-1, j)
                insert(mat, kij, idx(nx1, i, j, 2), vy3_coeff*v_scale) #vy3 = vy(i, j)
                insert(mat, kij, idx(nx1, i+1, j, 2), vy4_coeff*v_scale) #vy4 = vy(i+1, j)
                insert(mat, kij, idx(nx1, i, j+1, 2), vy5_coeff*v_scale) #vy5 = vy(i, j+1)

                #vx coefficients
                vx1_coeff = eta1/(dx*dy)
                vx2_coeff = -eta1/(dx*dy)
                vx3_coeff = -eta2/(dx*dy)
                vx4_coeff = eta2/(dx*dy)

                # Store vx coefficients
                insert(mat, kij, idx(nx1, i, j-1, 1), vx1_coeff*v_scale) #vx1 = vx(i, j-1)
                insert(mat, kij, idx(nx1, i+1, j-1, 1), vx2_coeff*v_scale) #vx2 = vx(i+1, j-1)
                insert(mat, kij, idx(nx1, i, j, 1), vx3_coeff*v_scale) #vx3 = vx(i, j)
                insert(mat, kij, idx(nx1, i+1, j, 1), vx4_coeff*v_scale) #vx4 = vx(i+1, j)

                # -dP/dy
                insert(mat, kij, idx(nx1, i+1, j, 0), -p_scale/dy)
                insert(mat, kij, idx(nx1, i, j, 0), p_scale/dy)
                
                # RHS
                b[kij] = -gy*rho[i, j]
    return mat[0], mat[1], mat[2], b