"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: model.py
Description: This file implements basic Stokes flow and continuity equations
             for boyancy-driven flow with marker-in-cell method.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numba as nb
import numpy as np

from Pyroclast.model.base_model import BaseModel
from Pyroclast.interpolation.linear_2D_cpu \
    import interpolate_markers2grid as interpolate
from Pyroclast.profiling import timer

from Pyroclast.solvers.stokes_2d import UzawaSolver
from Pyroclast.model.stokes_2D import IncompressibleStokes2D


# Model class
class IncompressibleStokes2DMG(IncompressibleStokes2D): # Inherit from BaseModel
                                     # this automatically does some magic
    """
    Basic Stokes flow and continuity equations for buoyancy-driven flow in 2D.
    
    This class solves the Stokes and continuity equations for a fluid on a 2D
    domain.

    The domain of choice is a rectangular box with a circular inclusion of
    different viscosity and density. 

    We use a basic 2D staggered grid with uniform spacing in each dimension.
    """
    def __init__(self, ctx):
        # Initialize the model
        super().__init__(ctx)
        
        # Read context
        s, p, o = ctx
        
        # Interpolate density to pressure nodes
        # This is needed to compute hydrostatic pressure
        # Only needed for the first iteration
        rhop = self.interpolate_rhop(ctx)

        # Set up initial guess for pressure
        s.p = compute_hydrostatic_pressure(s.nx1, s.ny1, s.dy,
                                           rhop, p.gy,
                                           p.p_ref,
                                           s.p)
        
        # Store rhs arrays for problem
        self.p_rhs = np.zeros((s.ny1, s.nx1))
        self.vx_rhs = np.zeros((s.ny1, s.nx1))
        self.vy_rhs = np.zeros((s.ny1, s.nx1))

        self.p_res = np.zeros((s.ny1, s.nx1))
        self.vx_res = np.zeros((s.ny1, s.nx1))
        self.vy_res = np.zeros((s.ny1, s.nx1))

    def interpolate_rhop(self, ctx):
        # Read the context
        s, p, o = ctx

        return interpolate(s.xp,                      # Density on y-velocity nodes
                           s.yp,  
                           s.xm,                      # Marker x positions
                           s.ym,                      # Marker y positions
                           s.rhom,                    # Density
                           indexing="equidistant",    # Equidistant grid spacing
                           return_weights=False)      # Do not return weights

    def solve(self, ctx):
        # Read the context
        s, p, o = ctx

        # Recompute vy rhs
        self.vy_rhs[...] = -self.gy * s.rho

        # Create Uzawa solver
        # Determine scaling such that coarsest grid is ~30x30
        # Assume square gri
        # n / scaling ^ 6 = 30
        # => scaling = 1 / (30 / n)^(1/6)
        scaling = 1/((30.0 / max(s.nx1 - 1, s.ny1 - 1))**(1/5))
        solver = UzawaSolver(ctx, nlevels=6, scaling=scaling)

        # Solve the system
        s.p[...], s.vx[...], s.vy[...] = \
        solver.solve(self.p_rhs, self.vx_rhs, self.vy_rhs,
                     p_guess=s.p, vx_guess=s.vx, vy_guess=s.vy)
        
        # Solve with iterative refinement (optional)
        # s.p, s.vx, s.vy = \
        # refinement.run(s.p, s.vx, s.vy,
        #                self.p_rhs, self.vx_rhs, self.vy_rhs,
        #                p0=s.p, vx0=s.vx, vy0=s.vy,
        #                max_refine=5, tol=1e-6)



@nb.njit(cache=True)
def compute_hydrostatic_pressure(nx1, ny1, dy, rho, gy, p_ref, p):
    """
    Compute the hydrostatic pressure field in-place, returning the modified array.
    """
    # Set pressure at the top boundary
    for j in range(1, nx1-1):
        p[1, j] = p_ref

    # Compute pressure field
    for i in range(2, ny1-1):
        for j in range(1, nx1-1):
            p[i, j] = p[i-1, j] + gy * dy * (rho[i, j] + rho[i-1, j]) / 2

    return p