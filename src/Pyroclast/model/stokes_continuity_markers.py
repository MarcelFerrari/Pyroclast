"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: stokes+continuity_markers.py
Description: This file implements basic Stokes flow and continuity equations
             for boyancy-driven flow with marker-in-cell method.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

See LICENSE file in the project root for full license information.
"""
import numba as nb
import numpy as np

from Pyroclast.linalg import xp
from Pyroclast.model.base_model import BaseModel
from Pyroclast.interpolation.linear_2D_cpu \
    import interpolate_markers2grid as interpolate
from Pyroclast.profiling import timer

# Initialize solver
from phasma import SparseLUD as SparseLU, CCSDSpmat as SparseMatrix
solver = SparseLU()

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

        self.rho = interpolate(self.ctx.grid.xrho_vy,      # Density on y-velocity nodes
                                self.ctx.grid.yrho_vy, 
                                self.ctx.pool.xm,           # Marker x positions
                                self.ctx.pool.ym,           # Marker y positions
                                (self.ctx.pool.rhom,),      # Marker density
                                indexing="equidistant",     # Equidistant grid spacing
                                return_weights=False)       # Do not return weights
        
        self.etab = interpolate(self.ctx.grid.xeta_b,      # Basic viscosity on grid nodes
                                 self.ctx.grid.yeta_b,
                                 self.ctx.pool.xm,          # Marker x positions
                                 self.ctx.pool.ym,          # Marker y positions
                                 (self.ctx.pool.etam,),     # Marker viscosity
                                 indexing="equidistant",    # Equidistant grid spacing
                                 return_weights=False)      # Do not return weights
        
        self.etap = interpolate(self.ctx.grid.xeta_p,      # Pressure viscosity on grid nodes
                                 self.ctx.grid.yeta_p,
                                 self.ctx.pool.xm,          # Marker x positions
                                 self.ctx.pool.ym,          # Marker y positions
                                 (self.ctx.pool.etam,),     # Marker viscosity
                                 indexing="equidistant",    # Equidistant grid spacing
                                 return_weights=False)      # Do not return weights
     
    def solve(self):
        # Assemble the matrix
        n_threads = nb.get_num_threads()
        mat, b = assemble(self.nx1, self.ny1, self.dx, self.dy,
                                         self.gy, self.etap, self.etab, self.rho,
                                         self.BC, n_threads)
        
        A = SparseMatrix()
        A.set_from_triplets(*mat)

        # Call spsolve with explicit types
        with timer.time_section("Model Solve", "spSolve"):
            solver.factorize(A, scale_matrix='full')
            x = solver.solve(b)

        # Print residual
        print("Residual: ", xp.linalg.norm(b - A * x)/xp.linalg.norm(b))

        # Extract solution quantities
        x = x.reshape((self.ny1, self.nx1, 3))
        self.p = x[:, :, 0]
        self.vx = x[:, :, 1]
        self.vy = x[:, :, 2]

        # Dump p, vx, vy to npz file
        if self.frame % self.dump_interval == 0:
            with open(f"frame_{str(self.frame).zfill(4)}.npz", 'wb') as f:
                xp.savez(f, p=self.p, vx=self.vx, vy=self.vy, rho=self.rho, etab=self.etab, etap=self.etap)
        self.frame += 1

    def finalize(self):
        # Write rho, eta_b, eta_p, vx, vy and p to npz file
        xp.savez("solution.npz", rho=self.rho, etab=self.etab, etap=self.etap,
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
def insert(mat, cur, i, j, v):
    # Mat is a tuple (i_idx, j_idx, vals)
    mat[0][cur] = i
    mat[1][cur] = j
    mat[2][cur] = v

    # Increment current index
    cur[0] += 1
    
@timer.time_function("Model Solve", "Assemble")
@nb.njit(parallel=False, cache=True)
def assemble(nx1, ny1, dx, dy, gy, etap, etab, rho, BC, n_threads):
    # Assemble matrix in COO format
    n_eqs = 3               # Number of equations to solve
    n_rows = nx1*ny1*n_eqs  # Number of rows in the matrix        
    max_nnz = 12            # Maximum number of non-zero elements (~12 per row)

    # Preallocate memory for COO format
    i_idx = np.zeros((max_nnz*n_rows,), dtype=xp.int32)
    j_idx = np.zeros((max_nnz*n_rows,), dtype=xp.int32)
    vals = np.zeros((max_nnz*n_rows,), dtype=xp.float64)
    mat = (i_idx, j_idx, vals)

    # Compute the number of non-zero elements per thread
    n_chunks_per_thread = (nx1 * ny1) // n_threads
    nnz_per_thread = n_chunks_per_thread * n_eqs * max_nnz
    b = np.zeros((n_rows,))

    # Scaled pressure
    pscale = 1.0

    # Loop over the grid
    for tidx in nb.prange(n_threads):
        # Track current non-zero index for each thread
        cur = np.array([tidx * nnz_per_thread], dtype=np.int64)
        
        # Compute thread boundaries
        start = tidx * n_chunks_per_thread
        end = (tidx + 1) * n_chunks_per_thread
        
        # Last thread needs to compute remaining chunks
        if tidx == n_threads - 1:
            end = nx1 * ny1

        # We fuse the nx and ny loops into a single loop
        # in order to significantly improve parallelism
        for chunk in nb.prange(start, end): 
            # Compute the i, j indices from the chunk index
            i = chunk // nx1
            j = chunk % nx1

            # Continuity equation (P)
            kij = idx(nx1, i, j, 0)

            # Set P = 0 for ghost nodes
            if i == 0 or j == 0 or i == ny1 - 1 or j == nx1 - 1:
                insert(mat, cur, kij, kij, 1.0)
                b[kij] = 0.0
            elif i == 1 and j == 1:
                insert(mat, cur, kij, kij, pscale)
                b[kij] = 1e9
            else:
                # vx coefficients
                insert(mat, cur, kij, idx(nx1, i, j, 1), 1/dx)
                insert(mat, cur, kij, idx(nx1, i, j-1, 1), -1/dx)

                # vy coefficients
                insert(mat, cur, kij, idx(nx1, i, j, 2), 1/dy)
                insert(mat, cur, kij, idx(nx1, i-1, j, 2), -1/dy)

                # RHS
                b[kij] = 0.0

            # 2) x-momentum equation (vx)
            kij = idx(nx1, i, j, 1)
            
            if j == 0 or j >= nx1-2: # Last two nodes in x-direction
                insert(mat, cur, kij, kij, 1.0)
                b[kij] = 0.0
            elif i == 0 or i == ny1 - 1: # First and last nodes in y-direction
                insert(mat, cur, kij, idx(nx1, i, j, 1), 1.0)
                if i == 0: # Top boundary
                    insert(mat, cur, kij, idx(nx1, i+1, j, 1), BC)
                else: # Bottom boundary
                    insert(mat, cur, kij, idx(nx1, i-1, j, 1), BC)
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
                insert(mat, cur, kij, idx(nx1, i, j-1, 1), vx1_coeff) #vx1 = vx(i, j-1)
                insert(mat, cur, kij, idx(nx1, i-1, j, 1), vx2_coeff) #vx2 = vx(i-1, j)
                insert(mat, cur, kij, idx(nx1, i, j, 1), vx3_coeff) #vx3 = vx(i, j)
                insert(mat, cur, kij, idx(nx1, i+1, j, 1), vx4_coeff) #vx4 = vx(i+1, j)
                insert(mat, cur, kij, idx(nx1, i, j+1, 1), vx5_coeff) #vx5 = vx(i, j+1)

                # vy coefficients
                vy1_coeff = eta1/(dx*dy)
                vy2_coeff = -eta2/(dx*dy)
                vy3_coeff = -eta1/(dx*dy)
                vy4_coeff = eta2/(dx*dy)

                #Store vy coefficients
                insert(mat, cur, kij, idx(nx1, i-1, j, 2), vy1_coeff) #vy1 = vy(i-1, j)
                insert(mat, cur, kij, idx(nx1, i, j, 2), vy2_coeff) #vy2 = vy(i, j)
                insert(mat, cur, kij, idx(nx1, i-1, j+1, 2), vy3_coeff) #vy3 = vy(i-1, j+1)
                insert(mat, cur, kij, idx(nx1, i, j+1, 2), vy4_coeff) #vy4 = vy(i, j+1)
                        
                # -dP/dx
                insert(mat, cur, kij, idx(nx1, i, j+1, 0), -pscale/dx)
                insert(mat, cur, kij, idx(nx1, i, j, 0), pscale/dx)
                
                # RHS
                b[kij] = 0.0


            # 3) y-momentum equation (vy)
            kij = idx(nx1, i, j, 2)

            if i == 0 or i >= ny1 - 2: # Last two nodes in y-direction
                insert(mat, cur, kij, kij, 1.0)
                b[kij] = 0.0
            elif j == 0 or j == nx1 - 1:
                insert(mat, cur, kij, kij, 1.0)
                if j == 0:
                    insert(mat, cur, kij, idx(nx1, i, j+1, 2), BC)
                else:
                    insert(mat, cur, kij, idx(nx1, i, j-1, 2), BC)
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
                insert(mat, cur, kij, idx(nx1, i, j-1, 2), vy1_coeff) #vy1 = vy(i, j-1)
                insert(mat, cur, kij, idx(nx1, i-1, j, 2), vy2_coeff) #vy2 = vy(i-1, j)
                insert(mat, cur, kij, idx(nx1, i, j, 2), vy3_coeff) #vy3 = vy(i, j)
                insert(mat, cur, kij, idx(nx1, i+1, j, 2), vy4_coeff) #vy4 = vy(i+1, j)
                insert(mat, cur, kij, idx(nx1, i, j+1, 2), vy5_coeff) #vy5 = vy(i, j+1)

                #vx coefficients
                vx1_coeff = eta1/(dx*dy)
                vx2_coeff = -eta1/(dx*dy)
                vx3_coeff = -eta2/(dx*dy)
                vx4_coeff = eta2/(dx*dy)

                # Store vx coefficients
                insert(mat, cur, kij, idx(nx1, i, j-1, 1), vx1_coeff) #vx1 = vx(i, j-1)
                insert(mat, cur, kij, idx(nx1, i+1, j-1, 1), vx2_coeff) #vx2 = vx(i+1, j-1)
                insert(mat, cur, kij, idx(nx1, i, j, 1), vx3_coeff) #vx3 = vx(i, j)
                insert(mat, cur, kij, idx(nx1, i+1, j, 1), vx4_coeff) #vx4 = vx(i+1, j)

                # -dP/dy
                insert(mat, cur, kij, idx(nx1, i+1, j, 0), -pscale/dy)
                insert(mat, cur, kij, idx(nx1, i, j, 0), pscale/dy)
                
                # RHS
                b[kij] = -gy*rho[i, j]
    return mat, b

