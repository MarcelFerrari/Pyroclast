"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: GMG.py
Description: This file implements coupled pressure-velocity multigrid solver based
             on Uzawa's iteration for the Stokes equations.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from Pyroclast.interpolation.linear_2D_cpu \
    import interpolate_markers2grid as interpolate

from Pyroclast.profiling import timer
import scipy.sparse as sp
import phasma as ph
import numpy as np
from .smoother import *
from .implicit_operators import *
from .utils import *
from .gmg_routines import *
from .direct import assemble_matrix
import matplotlib.pyplot as plt


# Class to store each grid level
class Grid:
    def __init__(self, shape, xmin, xmax, ymin, ymax, level, BC):
        self.level = level
        self.ny1 = shape[0]
        self.nx1 = shape[1]

        self.x = np.linspace(xmin, xmax, self.nx1)
        self.y = np.linspace(ymin, ymax, self.ny1)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # Create additional staggered grid nodes
        self.xvx = self.x
        self.yvx = self.y - self.dy/2
        self.xvy = self.x - self.dx/2
        self.yvy = self.y
        self.xp = self.x - self.dx/2
        self.yp = self.y - self.dy/2

        # Material properties
        self.etab = None # Computational viscosity in basic nodes
        self.etap = None # Computational viscosity in pressure nodes
        self.rho = None # True density in vy nodes

        # Pressure reference
        self.p_ref = None

        # Store local solution
        self.p = np.zeros((self.ny1, self.nx1))
        self.vx = np.zeros((self.ny1, self.nx1))
        self.vy = np.zeros((self.ny1, self.nx1))

        # Store local rhs
        self.p_rhs = None
        self.vx_rhs = None
        self.vy_rhs = None

        # Store local residual
        self.p_res = np.zeros((self.ny1, self.nx1))
        self.vx_res = np.zeros((self.ny1, self.nx1))
        self.vy_res = np.zeros((self.ny1, self.nx1))

        # Store matrix if needed
        self.mat = None

        # Boundary conditions
        self.BC = BC

    @timer.time_function("Model Solve", "Update Residual")
    def update_residual(self):
        # Update residual vectors in-place
        vx_residual(self.nx1,
                    self.ny1,
                    self.dx,self.dy,
                    self.etap, self.etab,
                    self.vx, self.vy, self.p,
                    self.vx_res, self.vx_rhs)
        
        vy_residual(self.nx1, self.ny1,
                    self.dx,self.dy,
                    self.etap, self.etab,
                    self.vx, self.vy, self.p,
                    self.vy_res, self.vy_rhs)
        
        p_residual(self.nx1, self.ny1,
                   self.dx, self.dy,
                   self.vx, self.vy,
                   self.p_res, self.p_rhs)
    
    @timer.time_function("Model Solve", "Compute Residual Norm")
    def residual_norm(self):
        # Average L2 residual over all cells
        p_res_norm = np.linalg.norm(self.p_res.reshape(-1))/(self.nx1*self.ny1)
        vx_res_norm = np.linalg.norm(self.vx_res.reshape(-1))/(self.nx1*self.ny1)
        vy_res_norm = np.linalg.norm(self.vy_res.reshape(-1))/(self.nx1*self.ny1)
        return p_res_norm, vx_res_norm, vy_res_norm
    
    def update_bc(self):
        apply_BC(self.p, self.vx, self.vy, self.BC)
            

class Multigrid:
    def __init__(self, ctx, levels, grid_scaling=2.0):
        self.ctx = ctx
        self.levels = levels
        self.grids = []

        # Grab domain limits directly from finest grid
        self.xmin = self.ctx.grid.x[0]
        self.xmax = self.ctx.grid.x[-1]
        self.ymin = self.ctx.grid.y[0]
        self.ymax = self.ctx.grid.y[-1]
        self.p_ref = self.ctx.params.p_ref
        self.grid_scaling = grid_scaling
        
        # Store some constants
        self.gy = self.ctx.model.gy
        self.BC = self.ctx.model.BC
        self.p_ref = self.ctx.params.p_ref
        self.relax_v = 0.9
        self.relax_p = 0.9

        self.solver = ph.direct.Eigen_SparseLU(ph.Scale.Full)

        # Initialize grids
        self.init_grids()

        
        # self.use_bicgstab = False
        # if self.use_bicgstab:
        #     self.bicg = ph.matfree.Eigen_BiCGSTAB()
        #     self.bicg.set_max_iterations(10)
        #     self.bicg.set_tolerance(1e-12)

    def make_grid(self, shape, level):
        return Grid(shape, self.xmin, self.xmax, self.ymin, self.ymax, level, self.ctx.model.BC)
    
    @timer.time_function("Model Solve", "Initialization")
    def init_grids(self):
        # Initialize finest level
        gh = self.make_grid((self.ctx.grid.ny1, self.ctx.grid.nx1), level=0)
        
        # Copy over material properties from the context
        gh.etab = self.ctx.model.etab
        gh.etap = self.ctx.model.etap
        gh.rho = self.ctx.model.rho

        # Init hydrostatic pressure
        gh.p_ref = self.p_ref
        gh.p = compute_hydrostatic_pressure(gh.nx1, gh.ny1, gh.dy, self.ctx.model.rhop, self.gy, gh.p_ref)
        
        # Init rhs arrays
        gh.vx_rhs = np.zeros((gh.ny1, gh.nx1))
        gh.vy_rhs = -self.gy * gh.rho
        gh.p_rhs = np.zeros((gh.ny1, gh.nx1))

        # Append to list of grids
        self.grids.append(gh)

        print(f"Grid 1: {gh.nx1} x {gh.ny1} – Fine grid")

        # Get grid size
        nx = gh.nx1-1
        ny = gh.ny1-1

        # Initialize coarser levels
        for l in range(1, self.levels):
            # Initialize new coarse level grid
            nx1 = int(nx / (self.grid_scaling**l)) + 1
            ny1 = int(ny / (self.grid_scaling**l)) + 1
            
            shape = (nx1, nx1)

            # Create grid
            gH = self.make_grid(shape, level=l)
            
            # Restrict the material properties from the fine grid to the coarse grid
            gH.rho = restrict(gh.yvx, gh.yvy, gh.rho,
                                gH.xvx, gH.yvx)
            gH.etab = restrict(gh.x, gh.y, gh.etab,
                                gH.x, gH.y)
            gH.etap = restrict(gh.xp, gh.yp, gh.etap,
                                gH.xp, gH.yp)
            
            # Coarsest level
            if l == self.levels - 1:
                data = assemble_matrix(gH.nx1, gH.ny1, gH.dx, gH.dy, gH.etap, gH.etab)
                gH.mat = ph.CCSMatrix(*data)
                self.solver.compute(gH.mat)

            # Append to list of grids
            self.grids.append(gH)
            print(f"Grid {l+1}: {gH.nx1} x {gH.ny1}")
            gh = gH

    @timer.time_function("Model Solve", "Smoothing")
    def smooth(self, g, max_iter):
        # print(g.level)
        # if g.mat is not None: # Use direct solver for coarsest level
        #     # Fix RHS conditions
        #     g.p_rhs[0, :] = 0.0
        #     g.p_rhs[-1, :] = 0.0
        #     g.p_rhs[:, 0] = 0.0
        #     g.p_rhs[:, -1] = 0.0

        #     g.vx_rhs[0, :] = 0.0
        #     g.vx_rhs[-1, :] = 0.0
        #     g.vx_rhs[:, 0] = 0.0
        #     g.vx_rhs[:, -2:] = 0.0

        #     g.vy_rhs[0, :] = 0.0
        #     g.vy_rhs[-2:, :] = 0.0
        #     g.vy_rhs[:, 0] = 0.0
        #     g.vy_rhs[:, -1] = 0.0

        #     # Assemble rhs vector
        #     rhs = np.zeros((3, g.ny1, g.nx1))
        #     rhs[0, ...] = g.p_rhs
        #     rhs[1, ...] = g.vx_rhs
        #     rhs[2, ...] = g.vy_rhs       

        #     rhs = rhs.reshape(-1)
        #     u = self.solver.solve(rhs)
        
        #     u = u.reshape(3, g.ny1, g.nx1)
            
        #     # g.p = (1 - self.relax_p) * g.p + self.relax_p * u[0, ...]
        #     g.vx = (1 - self.relax_v) * g.vx + self.relax_v * u[1, ...]
        #     g.vy = (1 - self.relax_v) * g.vy + self.relax_v * u[2, ...]
            
        #     apply_vx_BC(g.vx, g.BC)
        #     apply_vy_BC(g.vy, g.BC)

        #     pressure_uzawa_sweep(g.nx1, g.ny1, g.dx, g.dy, g.vx, g.vy,
        #                         g.p, g.etap, self.relax_p, g.p_rhs)

        #     apply_p_BC(g.p)
        #     # apply_BC(g.p, g.vx, g.vy, g.BC)
        #     # Compute residual
        #     # g.update_residual()
        #     # print("level:", g.level, g.residual_norm())
        #     # print("Saving Direct")
        #     # np.savez(f"direct.npz", p = g.p, vx = g.vx, vy = g.vy, p_res = g.p_res, vx_res = g.vx_res, vy_res = g.vy_res)
        #     # exit()

            
        # else: # Use red-black Gauss-Seidel for coarse levels
        red_black_gs(g.nx1, g.ny1,
                    g.dx, g.dy,
                    g.etap, g.etab,
                    g.vx, g.vy, g.p,
                    self.relax_v, self.relax_p,
                    g.p_ref, g.BC, g.p_rhs, g.vx_rhs, g.vy_rhs,
                    max_iter)#, chunk_size=32)
        
            # if g.mat is not None:
            #     print("Saving Smooth")
            #     np.savez(f"smooth.npz", p = g.p, vx = g.vx, vy = g.vy, p_res = g.p_res, vx_res = g.vx_res, vy_res = g.vy_res)
            #     exit()

        # else: # Use BiCGSTAB for the finest level
        #     for _ in range(10):
        #         # Build rhs vector for velocity assuming constant pressure
        #         rhs = assemble_momentum_rhs(g.nx1, g.ny1, g.dx, g.dy, g.rho, self.gy, g.p)

        #         rows = 2 * g.ny1 * g.nx1
        #         # Allocate memory for BC fix
        #         u_bc = np.zeros((g.ny1, g.nx1, 2))

        #         # Build operator
        #         def closure(u, u_new):
        #             # Reshape u and u_new
        #             u = u.reshape(g.ny1, g.nx1, 2)
        #             u_new = u_new.reshape(g.ny1, g.nx1, 2)

        #             # Copy current solution to u_bc and apply boundary conditions
        #             u_bc[...] = u[...]
        #             apply_u_BC(u_bc, g.BC)

        #             # Apply momentum operator
        #             momentum_operator(u_bc, g.nx1, g.ny1, g.dx, g.dy, g.etap, g.etab, g.BC, u_new)
                
        #         # Copy current solution to u
        #         guess = np.zeros((g.ny1, g.nx1, 2))
        #         guess[..., 0] = g.vx
        #         guess[..., 1] = g.vy

        #         u_new = self.bicg.solve(closure, rows, rows, rhs.reshape(-1), guess.reshape(-1))
        #         u_new = u_new.reshape(g.ny1, g.nx1, 2)

        #         # print("BiCGSTAB error: ", self.bicg.error(), " iterations: ", self.bicg.iterations())
                
        #         # Update velocity
        #         g.vx = (1 - self.relax_v) * g.vx + self.relax_v * u_new[..., 0]
        #         g.vy = (1 - self.relax_v) * g.vy + self.relax_v * u_new[..., 1]

        #         # Fix boundary conditions
        #         apply_vx_BC(g.vx, g.BC)
        #         apply_vy_BC(g.vy, g.BC)

        #         # Pressure update
        #         pressure_uzawa_sweep(g.nx1, g.ny1, g.dx, g.dy, g.vx, g.vy,
        #                                 g.p, g.etap, self.relax_p, g.p_rhs, p_ref = g.p_ref)
        #         pressure_uzawa_sweep(g.nx1, g.ny1, g.dx, g.dy, g.vx, g.vy,
        #                                 g.p, g.etap, self.relax_p, g.p_rhs, p_ref = g.p_ref)

        #         # Fix pressure boundary conditions
        #         apply_p_BC(g.p)

    def vcycle(self, level, nu1, nu2, gamma):
        # Extract grid information
        gh = self.grids[level] # Fine grid

        # Pre-smoothing: Perform nu1 smoothing steps on the fine grid
        if nu1 > 0:
            self.smooth(gh, max_iter=nu1)
        
        # Solve the coarse grid problem using recursive V-cycle
        if level + 1 < self.levels:
            gH = self.grids[level+1] # Coarse grid
            
            # Update residuals on the fine grid
            gh.update_residual()
            
            # Restrict the residual to the coarse grid
            # Pressure must be scaled by viscosity to get a smoother result
            rH_p = restrict(gh.xp, gh.yp, gh.p_res*gh.etap, gH.xp, gH.yp)
            rH_p /= gH.etap
            rH_vx = restrict(gh.xvx, gh.yvx, gh.vx_res, gH.xvx, gH.yvx)
            rH_vy = restrict(gh.xvy, gh.yvy, gh.vy_res, gH.xvy, gH.yvy)

            # Overwrite the rhs on the coarse grid
            gH.p_rhs = rH_p
            gH.vx_rhs = rH_vx
            gH.vy_rhs = rH_vy

            for _ in range(gamma):
                self.vcycle(level+1, 2*nu1, 2*nu2, gamma)

            # Interpolate the correction to the fine grid
            uH_p = gH.p
            uH_vx = gH.vx
            uH_vy = gH.vy
            
            # Interpolate the correction to the fine grid
            eh_p = prolong(gh.xp, gh.yp, gH.xp, gH.yp, uH_p*gH.etap)
            eh_p /= gh.etap
            eh_vx = prolong(gh.xvx, gh.yvx, gH.xvx, gH.yvx, uH_vx)
            eh_vy = prolong(gh.xvy, gh.yvy, gH.xvy, gH.yvy, uH_vy)
            
            # Update the solution on the fine grid
            gh.p += eh_p
            gh.vx += eh_vx
            gh.vy += eh_vy

            # Apply boundary conditions
            if level == 0:
                gh.update_bc()

            # Reset coarse grid solution
            gH.p[:, :] = 0.0
            gH.vx[:, :] = 0.0
            gH.vy[:, :] = 0.0
        
        # Post-smoothing: Perform nu2 smoothing steps on the fine grid
        if nu2 > 0:
            self.smooth(gh, max_iter=nu2)

        # Update residuals
        gh.update_residual()

        return gh.residual_norm() if level == 0 else None

    def solve(self, max_cycles = 50, tol = 1e-4, nu1=3, nu2=3, gamma=1, pre_smooth=0, p_guess = None, vx_guess=None, vy_guess=None):
        
        if p_guess is not None:
            self.grids[0].p[...] = p_guess[...]

        if vx_guess is not None:
            self.grids[0].vx[...] = vx_guess[...]

        if vy_guess is not None:
            self.grids[0].vy[...] = vy_guess[...]

        if pre_smooth > 0:
            self.smooth(self.grids[0], max_iter=pre_smooth)

        for c in range(max_cycles):
            print("Cycle: ", c)

            # Perform V-cycles
            p_res, vx_res, vy_res = self.vcycle(0, nu1, nu2, gamma)

            print("Continuity residual: ", p_res)
            print("X-momentum residual: ", vx_res)
            print("Y-momentum residual: ", vy_res)

            # Check convergence
            if max(p_res, vx_res, vy_res) < tol:
                break

        return self.grids[0].p, self.grids[0].vx, self.grids[0].vy