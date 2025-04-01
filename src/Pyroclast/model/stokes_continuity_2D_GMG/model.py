"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: model.py
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

from .smoother import *
from .implicit_operators import vx_residual, vy_residual, p_residual
from .utils import *
from .gmg import Multigrid

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
        
        self.rhop = interpolate(self.ctx.grid.xp,           # Density on pressure nodes
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
        solver = Multigrid(self.ctx, 6, grid_scaling=2.0)    
        self.p, self.vx, self.vy = solver.solve(max_cycles=100, tol=0.0, nu1 = 5, nu2 = 5, gamma=1,
                                                p_guess = self.p, vx_guess = self.vx, vy_guess = self.vy)
        self.dump_frame()

    def dump_frame(self):
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

