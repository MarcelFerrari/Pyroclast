"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: advection.py
Description: this file contains the implementation of a simple
             constant velocity advection model in 2D.
             This is meant to test marker advection.
             

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
import numba as nb
import numpy as np
from Pyroclast.model.base_model import BaseModel
from Pyroclast.profiling import timer


# Model class
class ConstantVelocityAdvection2D(BaseModel):
    """
    Constant velocity advection model in 2D.
     
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
        s.rho = np.zeros((p.ny, p.nx))
        s.etab = np.zeros((p.ny, p.nx))
        s.etap = np.zeros((p.ny, p.nx))
        
        # Set up variables for solution quantities
        # These are defined on ghost nodes to enforce the boundary conditions
        Y, X = np.meshgrid(s.y, s.x, indexing='ij')

        # Set up the velocity field
        s.vx = (-(Y/p.ysize - 0.5)) * 1e-7
        s.vy = (X/p.xsize - 0.5) * 1e-7

        # Set up time step
        vxmax = np.max(np.abs(s.vx))
        vymax = np.max(np.abs(s.vy))
        dty = p.cfl_dispmax * s.dy / vymax
        dtx = p.cfl_dispmax * s.dx / vxmax
        self.dt = min(dtx, dty)

        assert s.vx.shape == (s.ny1, s.nx1), "vx shape mismatch"
        assert s.vy.shape == (s.ny1, s.nx1), "vy shape mismatch"


    def update_time_step(self, ctx):
        # Read the context
        s, p, o = ctx
        s.dt = self.dt
        
    def solve(self, ctx):
        # Nothing to do here
        pass