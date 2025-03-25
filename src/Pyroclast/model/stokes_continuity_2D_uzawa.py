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

from Pyroclast.linalg import xp
from Pyroclast.model.base_model import BaseModel
from Pyroclast.interpolation.linear_2D_cpu \
    import interpolate_markers2grid as interpolate
from Pyroclast.profiling import timer
import matplotlib.pyplot as plt

# Initialize solver
import phasma as ph

# solver = ph.SparseLU(ph.ScalingType.FULL)

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

        # Scaling
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
        
        self.rhop = interpolate(self.ctx.grid.xp,           # Density on y-velocity nodes
                                self.ctx.grid.yp, 
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
        solver = UzawaSolver()

        with timer.time_section("Model Solve", "Iterative Solve"):
            # Initialize solutions
            p = compute_hydrostatic_pressure(self.nx1, self.ny1, self.dy, self.rhop, self.gy, self.p_ref, self.p_scale)
            v = np.zeros((self.ny1, self.nx1, 2))
            
            # Assemble rhs
            p_rhs, v_rhs = assemble_rhs(self.nx1, self.ny1, self.gy, self.rho, self.p_ref, self.p_scale)
            self.p, self.vx, self.vy, p_res, vx_res, vy_res = solver.solve(self.nx1, self.ny1, self.dx, self.dy,
                                                                            p, v, p_rhs, v_rhs,
                                                                            self.etab, self.etap, self.rho, self.rhop,
                                                                            self.p_ref, self.p_scale, self.v_scale,
                                                                            self.gy, self.BC)

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

class UzawaSolver:
    def __init__(self, max_iter = 100, p_relax = 0.7, v_relax = 0.7):
        self.max_iter = max_iter
        self.p_relax = p_relax
        self.v_relax = v_relax

    def solve(self, nx1, ny1, dx, dy, p, v, p_rhs, v_rhs, etab, etap, rho, rhop, p_ref, p_scale, v_scale, gy, BC):
        solver = ph.iterative.Eigen_BiCGSTAB()
        solver.set_max_iterations(200)
        solver.set_tolerance(1e-12)

        # Initialize the work arrays
        p_wrk = np.zeros((ny1, nx1)) # Pressure work array        
        v_wrk = np.zeros((ny1, nx1, 2)) # Velocity work array

        # Pressure-mass scaling factors
        beta = etap/p_scale

        for i in range(self.max_iter):

            # 1) Solve momentum equations
            data, rhs = assemble_momentum(nx1, ny1, dx, dy, etap, etab, rho, gy, p, p_scale, v_scale, BC)
            A = ph.CRSMatrix(*data)
            
            # Solve for new velocity field
            v_new = solver.solve(A, rhs, v.reshape(-1))
            
            # Reshape to grid shape
            v_new = v_new.reshape((ny1, nx1, 2))

            # Apply under-relaxation
            v = (1 - self.v_relax)*v + self.v_relax*v_new

            apply_velocity_bc(v, BC)
            
            # 2) Apply Uzawa pressure correction
            # Update pressure
            apply_pressure_update(nx1, ny1, dx, dy, p, p_wrk, v, p_ref, beta, self.p_relax, p_scale, v_scale, max_iter=2)

            # 3) Compute residuals
            p_res, vx_res, vy_res = residual(p, v, p_wrk, v_wrk, p_rhs, v_rhs, nx1, ny1, dx, dy, etap, etab, p_scale, v_scale, BC)
            print(f"Iteration {i}: {p_res:.4e}, {vx_res:.4e}, {vy_res:.4e}")

            if max(p_res, vx_res, vy_res) < 1e-3:
                break

        return p, v[:, :, 0], v[:, :, 1], p_res, vx_res, vy_res

# ================= Matrix-free Operators =================
@nb.njit(cache=True)
def compute_hydrostatic_pressure(nx1, ny1, dy, rho, gy, p_ref, p_scale):
    """
    Compute the hydrostatic pressure field.
    """
    p = np.zeros((ny1, nx1))

    # Set pressure at the top boundary
    for i in range(1, nx1-1):
        p[1, i] = p_ref/p_scale

    # Compute pressure field
    for i in range(2, ny1-1):
        for j in range(1, nx1-1):
            p[i, j] = p[i-1, j] + gy * dy * (rho[i, j] + rho[i-1, j])/2 * 1/p_scale
    
    return p

# Numba compiled functions - Not part of the class
@nb.njit(cache = True)
def idx(nx1, i, j, q):
    # Helper function to map 2D indices to 1D index
    # i: matrix row index (y-index)
    # j: matrix column index (x-index)
    # q: variable index (0: vx, 1: vy)
    return 2*(i*nx1 + j) + q
    
@nb.njit(cache = True)
def insert(mat, cur, i, j, v):
    # Mat is a tuple (i_idx, j_idx, vals)
    mat[0][cur] = i
    mat[1][cur] = j
    mat[2][cur] = v

    # Increment current index
    cur[0] += 1

@nb.njit(cache = True)
def assemble_momentum(nx1, ny1, dx, dy, etap, etab, rho, gy, p, p_scale, v_scale, BC):
    n_eqs = 2
    rows = nx1*ny1*n_eqs
    max_nnz = 15

    # Preallocate memory for COO format
    i_idx = np.zeros((max_nnz*rows,), dtype=np.int32)
    j_idx = np.zeros((max_nnz*rows,), dtype=np.int32)
    vals = np.zeros((max_nnz*rows,), dtype=np.float64)
    mat = (i_idx, j_idx, vals)
    cur = np.array([0], dtype=np.int64)

    # Preallocate memory for RHS
    b = np.zeros((nx1*ny1*n_eqs,), dtype=np.float64)
    for i in range(ny1):
        for j in range(nx1):
            # 2) x-momentum equation (vx)
            kij = idx(nx1, i, j, 0)
            
            if j == 0 or j >= nx1-2: # Last two nodes in x-direction
                insert(mat, cur, kij, kij, 1.0)
                b[kij] = 0.0
            elif i == 0 or i == ny1 - 1: # First and last nodes in y-direction
                insert(mat, cur, kij, idx(nx1, i, j, 0), v_scale)
                if i == 0: # Top boundary
                    insert(mat, cur, kij, idx(nx1, i+1, j, 0), BC*v_scale)
                else: # Bottom boundary
                    insert(mat, cur, kij, idx(nx1, i-1, j, 0), BC*v_scale)
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
                insert(mat, cur, kij, idx(nx1, i, j-1, 0), vx1_coeff*v_scale) #vx1 = vx(i, j-1)
                insert(mat, cur, kij, idx(nx1, i-1, j, 0), vx2_coeff*v_scale) #vx2 = vx(i-1, j)
                insert(mat, cur, kij, idx(nx1, i, j, 0), vx3_coeff*v_scale) #vx3 = vx(i, j)
                insert(mat, cur, kij, idx(nx1, i+1, j, 0), vx4_coeff*v_scale) #vx4 = vx(i+1, j)
                insert(mat, cur, kij, idx(nx1, i, j+1, 0), vx5_coeff*v_scale) #vx5 = vx(i, j+1)

                # vy coefficients
                vy1_coeff = eta1/(dx*dy)
                vy2_coeff = -eta2/(dx*dy)
                vy3_coeff = -eta1/(dx*dy)
                vy4_coeff = eta2/(dx*dy)

                #Store vy coefficients
                insert(mat, cur, kij, idx(nx1, i-1, j, 1), vy1_coeff*v_scale)   #vy1 = vy(i-1, j)
                insert(mat, cur, kij, idx(nx1, i, j, 1), vy2_coeff*v_scale)     #vy2 = vy(i, j)
                insert(mat, cur, kij, idx(nx1, i-1, j+1, 1), vy3_coeff*v_scale) #vy3 = vy(i-1, j+1)
                insert(mat, cur, kij, idx(nx1, i, j+1, 1), vy4_coeff*v_scale)   #vy4 = vy(i, j+1)
                        
                # RHS
                b[kij] = p_scale * (p[i, j+1] - p[i, j])/dx

            # 3) y-momentum equation (vy)
            kij = idx(nx1, i, j, 1)

            if i == 0 or i >= ny1 - 2: # Last two nodes in y-direction
                insert(mat, cur, kij, kij, 1.0)
                b[kij] = 0.0
            elif j == 0 or j == nx1 - 1:
                insert(mat, cur, kij, kij, v_scale)
                if j == 0:
                    insert(mat, cur, kij, idx(nx1, i, j+1, 1), BC*v_scale)
                else:
                    insert(mat, cur, kij, idx(nx1, i, j-1, 1), BC*v_scale)
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
                insert(mat, cur, kij, idx(nx1, i, j-1, 1), vy1_coeff*v_scale) #vy1 = vy(i, j-1)
                insert(mat, cur, kij, idx(nx1, i-1, j, 1), vy2_coeff*v_scale) #vy2 = vy(i-1, j)
                insert(mat, cur, kij, idx(nx1, i, j, 1), vy3_coeff*v_scale) #vy3 = vy(i, j)
                insert(mat, cur, kij, idx(nx1, i+1, j, 1), vy4_coeff*v_scale) #vy4 = vy(i+1, j)
                insert(mat, cur, kij, idx(nx1, i, j+1, 1), vy5_coeff*v_scale) #vy5 = vy(i, j+1)

                #vx coefficients
                vx1_coeff = eta1/(dx*dy)
                vx2_coeff = -eta1/(dx*dy)
                vx3_coeff = -eta2/(dx*dy)
                vx4_coeff = eta2/(dx*dy)

                # Store vx coefficients
                insert(mat, cur, kij, idx(nx1, i, j-1, 0), vx1_coeff*v_scale) #vx1 = vx(i, j-1)
                insert(mat, cur, kij, idx(nx1, i+1, j-1, 0), vx2_coeff*v_scale) #vx2 = vx(i+1, j-1)
                insert(mat, cur, kij, idx(nx1, i, j, 0), vx3_coeff*v_scale) #vx3 = vx(i, j)
                insert(mat, cur, kij, idx(nx1, i+1, j, 0), vx4_coeff*v_scale) #vx4 = vx(i+1, j)
                
                # RHS
                b[kij] = -gy*rho[i, j] + p_scale * (p[i+1, j] - p[i, j])/dy
    return mat, b

@nb.njit(cache=True)
def apply_velocity_bc(u, BC):
    # vx boundary conditions
    u[0, :, 0] = - BC * u[1, :, 0]   # Top boundary
    u[-1, :, 0] = - BC * u[-2, :, 0] # Bottom boundary
    u[:, 0, 0] = 0.0                 # Left boundary
    u[:, -2:, 0] = 0.0               # Right boundary

    # vy boundary conditions
    u[:, 0, 1] = - BC * u[:, 1, 1]   # Left boundary
    u[:, -1, 1] = - BC * u[:, -2, 1] # Right boundary
    u[0, :, 1] = 0.0                 # Top boundary
    u[-2:, :, 1] = 0.0               # Bottom boundary

@nb.njit(cache=True, parallel=True)
def continuity_residual(nx1, ny1, dx, dy, v, rax, v_scale):
    for i in nb.prange(1, ny1 - 1):
        for j in nb.prange(1, nx1 - 1):
            rax[i, j] = -v_scale*((v[i, j, 0] - v[i, j - 1, 0])/dx + (v[i, j, 1] - v[i - 1, j, 1])/dy)
    return rax

@nb.njit(cache=True, parallel=True)
def apply_pressure_update(nx1, ny1, dx, dy, p, p_wrk, v, p_ref, beta, relax_p, p_scale, v_scale, max_iter=1):
    # Compute continuity residual
    for _ in range(max_iter):
        # Residual is stored in p_wrk
        continuity_residual(nx1, ny1, dx, dy, v, p_wrk, v_scale)

        # Pressure update only on interior nodes
        for i in nb.prange(1, ny1-1): 
            for j in nb.prange(1, nx1-1):
                p[i, j] += p_wrk[i, j]*beta[i, j]*relax_p

        # Set Zero-pressure on the boundaries
        p[0, :] = 0.0
        p[-1, :] = 0.0
        p[:, 0] = 0.0
        p[:, -1] = 0.0

    # Apply pressure anchor point
    dp = p_ref/p_scale - p[1, 1]
    p += dp

# Distributed pressure update – does not work very well
# @nb.njit(cache=True, parallel=True)
# def apply_pressure_update(nx1, ny1, dx, dy, p, p_wrk, v, p_ref, beta, relax_p, p_scale, v_scale, max_iter=1):
#     for _ in range(max_iter):
#         # Step 1: Compute continuity residual r = ∇·v
#         continuity_residual(nx1, ny1, dx, dy, v, p_wrk, v_scale)  # stored in p_wrk

#         # Step 2: Distribute residual to pressure field
#         for i in nb.prange(2, ny1 - 2):
#             for j in nb.prange(2, nx1 - 2):
#                 # Grab residual and scaling
#                 res = p_wrk[i, j]
#                 b = beta[i, j] * relax_p
#                 update = res * b

#                 # Distribute to neighbors
#                 p[i, j]     +=  4.0 * update
#                 p[i+1, j]   += -1.0 * update
#                 p[i-1, j]   += -1.0 * update
#                 p[i, j+1]   += -1.0 * update
#                 p[i, j-1]   += -1.0 * update

#         # Step 3: Enforce boundary condition (zero-pressure on boundaries)
#         for j in nb.prange(nx1):
#             p[0, j] = 0.0
#             p[-1, j] = 0.0
#         for i in nb.prange(ny1):
#             p[i, 0] = 0.0
#             p[i, -1] = 0.0

#     # Step 4: Apply pressure anchor
#     dp = p_ref/p_scale - p[1, 1]
#     p += dp

@nb.njit(cache=True, parallel=True)
def operator(v, p, v_new, p_new, nx1, ny1, dx, dy, etap, etab, p_scale, v_scale, BC):
    """
    Compute implicit matrix-vector product A*u where A is the FD discretization matrix.
    """

    # Loop over the grid
    for i in nb.prange(ny1):
        for j in nb.prange(nx1):
            # Continuity equation (P)
            # Set P = 0 for ghost nodes
            if i == 0 or j == 0 or i == ny1 - 1 or j == nx1 - 1:
                p_new[i, j] = p[i, j] * p_scale
            elif i == 1 and j == 1:
                p_new[i, j] = p[i, j] * p_scale
            else:
                p_new[i, j] = (v_scale/dx) * v[i, j, 0] + \
                              (-v_scale/dx) * v[i, j-1, 0] + \
                              (v_scale/dy) * v[i, j, 1] + \
                              (-v_scale/dy) * v[i-1, j, 1]

            # 2) x-momentum equation (vx)
            if j == 0 or j >= nx1-2: # Last two nodes in x-direction
                v_new[i, j, 0] = v[i, j, 0]
            elif i == 0: # Top boundary
                v_new[i, j, 0] = v_scale * v[i, j, 0] + \
                                 (BC * v_scale) * v[i+1, j, 0]
            elif i == ny1 - 1: # First and last nodes in y-direction
                v_new[i, j, 0] = v_scale * v[i, j, 0] + \
                                 (BC * v_scale) * v[i-1, j, 0]
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

                # vy coefficients
                vy1_coeff = eta1/(dx*dy)
                vy2_coeff = -eta2/(dx*dy)
                vy3_coeff = -eta1/(dx*dy)
                vy4_coeff = eta2/(dx*dy)

                v_new[i, j, 0] = (vx1_coeff*v_scale) * v[i, j-1, 0] + \
                                 (vx2_coeff*v_scale) * v[i-1, j, 0] + \
                                 (vx3_coeff*v_scale) * v[i, j, 0] + \
                                 (vx4_coeff*v_scale) * v[i+1, j, 0] + \
                                 (vx5_coeff*v_scale) * v[i, j+1, 0] + \
                                 (vy1_coeff*v_scale) * v[i-1, j, 1] + \
                                 (vy2_coeff*v_scale) * v[i, j, 1] + \
                                 (vy3_coeff*v_scale) * v[i-1, j+1, 1] + \
                                 (vy4_coeff*v_scale) * v[i, j+1, 1] + \
                                 (-p_scale/dx) * p[i, j+1] + \
                                 (p_scale/dx)  * p[i, j]

            # 3) y-momentum equation (vy)
            if i == 0 or i >= ny1 - 2: # Last two nodes in y-direction
                v_new[i, j, 1] = v[i, j, 1]
            elif j == 0:
                v_new[i, j, 1] = v_scale * v[i, j, 1] + \
                                 (BC * v_scale) * v[i, j+1, 1]
            elif j == nx1 - 1:
               v_new[i, j, 1] = v_scale * v[i, j, 1] + \
                                 (BC * v_scale) * v[i, j-1, 1]
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

                #vx coefficients
                vx1_coeff = eta1/(dx*dy)
                vx2_coeff = -eta1/(dx*dy)
                vx3_coeff = -eta2/(dx*dy)
                vx4_coeff = eta2/(dx*dy)


                v_new[i, j, 1] = (vy1_coeff*v_scale) * v[i, j-1, 1] + \
                                 (vy2_coeff*v_scale) * v[i-1, j, 1] + \
                                 (vy3_coeff*v_scale) * v[i, j, 1] + \
                                 (vy4_coeff*v_scale) * v[i+1, j, 1] + \
                                 (vy5_coeff*v_scale) * v[i, j+1, 1] + \
                                 (vx1_coeff*v_scale) * v[i, j-1, 0] + \
                                 (vx2_coeff*v_scale) * v[i+1, j-1, 0] + \
                                 (vx3_coeff*v_scale) * v[i, j, 0] + \
                                 (vx4_coeff*v_scale) * v[i+1, j, 0] + \
                                 (-p_scale/dy) * p[i+1, j] + \
                                 (p_scale/dy)  * p[i, j]

@nb.njit(cache=True)
def assemble_rhs(nx1, ny1, gy, rho, p_ref, p_scale):
    p_rhs = np.zeros((ny1, nx1))
    v_rhs = np.zeros((ny1, nx1, 2), dtype=np.float64)
    
    # Pressure anchor point in (1, 1)
    p_rhs[1, 1] = p_ref/p_scale

    # y-momentum equation (vy)
    v_rhs[1:ny1-2, 1:nx1-1, 1] = -gy*rho[1:ny1-2, 1:nx1-1]

    return p_rhs, v_rhs

def residual(p, v, p_new, v_new, p_rhs, v_rhs, nx1, ny1, dx, dy, etap, etab, p_scale, v_scale, BC):
    # Compute matrix-vector product v_new = A*u
    operator(v, p, v_new, p_new, nx1, ny1, dx, dy, etap, etab, p_scale, v_scale, BC)
    omega = (nx1-1)*(ny1-1)
    p_res = np.linalg.norm((p_rhs - p_new).reshape(-1))/omega
    vx_res = np.linalg.norm((v_rhs[:, :, 0] - v_new[:, :, 0]).reshape(-1))/omega
    vy_res = np.linalg.norm((v_rhs[:, :, 1] - v_new[:, :, 1]).reshape(-1))/omega
    return p_res, vx_res, vy_res
