"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: basic2D.py
Description: This file implements basic marker pool operations for 2D staggered grids.
             with uniform spacing in each dimension.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np
import numba as nb
from random import uniform

from Pyroclast.pool.base_pool import BasePool
from Pyroclast.interpolation.linear_2D_cpu \
    import interpolate_grid2markers as interpolate
from logging import get_logger

logger = get_logger(__name__)

class Basic2DStokes(BasePool): # Inherit from BasePool
                               # this automatically does some magic
    """
        This class implements a basic pool of markers for 2D staggered grids.
        This is not meant to be used directly, but to be inherited by specific
        pool implementations.
    """
    def __init__(self, ctx):
        # The marker pool is always initialize after the grid
        # this means that any necessary grid information is already
        # available via the context.

        # Read context
        s, p, o = ctx
        
        # Compute the total number of markers
        # Save them in the simulation state
        s.nmx = (p.nx-1) * p.nmpcx
        s.nmy = (p.ny-1) * p.nmpcy
        s.nm = s.nmx * s.nmy

        # Compute marker spacing
        s.dxm = p.xsize/s.nmx
        s.dym = p.ysize/s.nmy

        # Create initial marker distribution
        # Marker positions
        s.xm = np.random.uniform(0, p.xsize, s.nm)
        s.ym = np.random.uniform(0, p.ysize, s.nm)
        
        # Marker velocities
        s.vxm = np.zeros(s.nm, dtype=np.float64)
        s.vym = np.zeros(s.nm, dtype=np.float64)

        # Print grid info
        self.info(ctx)

    
    def interpolate(self, ctx):
        """
        Interpolate velocity from grid to markers.
        We are solving simple stokes flow, so we only need to interpolate vx and vy.
        """
        s, p, o = ctx
        # Interpolate velocity from vx nodes of staggered grid to markers
        s.vxm = interpolate(s.xvx,                   # x and y coordinates of the vx nodes
                            s.yvx, 
                            s.xm, s.ym,              # x and y coordinates of the markers
                            (s.vx,),                 # tuple with the values to interpolate (vx)
                            indexing="equidistant")  # equidistant grid
        
        # Interpolate velocity from vy nodes of staggered grid to markers
        s.vym = interpolate(s.xvy,                   # x and y coordinates of the vy nodes
                            s.yvy,
                            s.xm, s.ym,              # x and y coordinates of the markers
                            (s.vy,),                 # tuple with the values to interpolate (vy)
                            indexing="equidistant")  # equidistant grid
                               
    def advect(self, ctx):
        s, p, o = ctx

        # Advect markers
        s.xm += s.vxm * s.dt
        s.ym += s.vym * s.dt

        # Apply periodic boundary conditions
        s.xm = np.mod(s.xm, p.xsize)
        s.ym = np.mod(s.ym, p.ysize)

    def info(self, ctx):
        s, p, o = ctx

        logger.info(10*"-" + " Marker Pool Info " + 10*"-")
        logger.info(f"Initialized {self.__class__.__name__} marker pool.")
        logger.info(f"Number of markers in x-direction (per cell): {p.nmpcx}")
        logger.info(f"Number of markers in y-direction (per cell): {p.nmpcy}")
        logger.info(f"Number of markers per cell: {p.nmpcx*p.nmpcy}")
        logger.info(f"Number of markers in x-direction: {s.nmx}")
        logger.info(f"Number of markers in y-direction: {s.nmy}")
        logger.info(f"Total number of markers: {s.nm}")
        logger.info(f"Marker spacing in x-direction: {s.dxm:.1f}")
        logger.info(f"Marker spacing in y-direction: {s.dym:.1f}")
        logger.info(39*"-")

class RK42DStokes(Basic2DStokes):
    """
    Same as Basic2DStokes, but implements RK4 advection scheme.
    """
    def interpolate_vx(self, xm, ym, ctx):
        """
        Shortcut function to interpolate vx velocity from grid to markers.
        """

        s, p, o = ctx
        return interpolate(s.xvx,       # x and y coordinates of the vx nodes
                           s.yvx, 
                           xm, ym,                  # x and y coordinates of the markers
                           (s.vx,),    # tuple with the values to interpolate (vx)
                           indexing="equidistant",  # equidistant grid
                           cont_corr="x")           # x-continuity correction  
    
    def interpolate_vy(self, xm, ym, ctx):
        """
        Shortcut function to interpolate vy velocity from grid to markers.
        """
        s, p, o = ctx
        # Interpolate velocity from vy nodes of staggered grid to markers
        return interpolate(s.xvy,       # x and y coordinates of the vy nodes
                           s.yvy,
                           xm, ym,                  # x and y coordinates of the markers
                           (s.vy,),    # tuple with the values to interpolate (vy)
                           indexing="equidistant",  # equidistant grid
                           cont_corr="y")           # y-continuity correction

    def advect(self, ctx):
        # Read context
        s, p, o = ctx

        #  ------- RK4 advection of markers -------
        #  Velocity in A = (x(m), y(m))
        xmA = s.xm
        ymA = s.ym
        
        def interpolate_vx(xmA, ymA):
            return self.interpolate_vx(xmA, ymA, ctx)
        def interpolate_vy(xmA, ymA):
            return self.interpolate_vy(xmA, ymA, ctx)
        
        vxmA = interpolate_vx(xmA, ymA)
        vymA = interpolate_vy(xmA, ymA)
        
        # Convenience variables
        dt = s.dt
        xsize = p.xsize
        ysize = p.ysize

        # Coordinates of B = (xA + 1/2 dt vxmA, yA + 1/2 dt vymA)
        xmB = np.mod(xmA + 1/2*dt*vxmA, xsize)
        ymB = np.mod(ymA + 1/2*dt*vymA, ysize)
        vxmB = interpolate_vx(xmB, ymB)
        vymB = interpolate_vy(xmB, ymB)

        # Coordinates of C = (xA + 1/2 dt vxmB, yA + 1/2 dt vymB)
        xmC = np.mod(xmA + 1/2*dt*vxmB, xsize)
        ymC = np.mod(ymA + 1/2*dt*vymB, ysize)
        vxmC = interpolate_vx(xmC, ymC)
        vymC = interpolate_vy(xmC, ymC)

        # Coordinates of D = (xA + dt vxmC, yA + dt vymC)
        xmD = np.mod(xmA + dt*vxmC, xsize)
        ymD = np.mod(ymA + dt*vymC, ysize)
        vxmD = interpolate_vx(xmD, ymD)
        vymD = interpolate_vy(xmD, ymD)

        vx_eff = (1/6)*(vxmA + 2*vxmB + 2*vxmC + vxmD)
        vy_eff = (1/6)*(vymA + 2*vymB + 2*vymC + vymD)

        # Update marker positions
        s.xm = np.mod(s.xm + dt*vx_eff, xsize)
        s.ym = np.mod(s.ym + dt*vy_eff, ysize)