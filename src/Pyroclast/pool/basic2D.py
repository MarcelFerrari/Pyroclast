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

class Basic2DStokes(BasePool): # Inherit from BasePool
                                      # this automatically does some magic
    """
        This class implements a basic pool of markers for 2D staggered grids.
        This is not meant to be used directly, but to be inherited by specific
        pool implementations.
    """
    def initialize(self):
        # The marker pool is always initialize after the grid
        # this means that any necessary grid information is already
        # available via the context.
        
        # Read number of markers from the context
        self.nmpcx = self.ctx.params.nmpcx # Number of markers per cell in x direction
        self.nmpcy = self.ctx.params.nmpcy # Number of markers per cell in y direction

        # Compute the total number of markers
        self.nmx = (self.ctx.params.nx-1) * self.nmpcx
        self.nmy = (self.ctx.params.ny-1) * self.nmpcy
        self.nm = self.nmx * self.nmy

        # Compute marker spacing
        self.dxm = self.ctx.params.xsize/self.nmx
        self.dym = self.ctx.params.ysize/self.nmy

        # Create initial marker distribution
        # Marker positions
        self.xm = np.zeros(self.nm, dtype=np.float64)
        self.ym = np.zeros(self.nm, dtype=np.float64)
        
        # Marker velocities
        self.vxm = np.zeros(self.nm, dtype=np.float64)
        self.vym = np.zeros(self.nm, dtype=np.float64)

        # Print grid info
        self.info()
    
    def interpolate(self):
        """
        Interpolate velocity from grid to markers.
        We are solving simple stokes flow, so we only need to interpolate vx and vy.
        """
        # Interpolate velocity from vx nodes of staggered grid to markers
        self.vxm = interpolate(self.ctx.grid.xvx,       # x and y coordinates of the vx nodes
                               self.ctx.grid.yvx, 
                               self.xm, self.ym,        # x and y coordinates of the markers
                               (self.ctx.model.vx,),    # tuple with the values to interpolate (vx)
                               indexing="equidistant")  # equidistant grid
        
        # Interpolate velocity from vy nodes of staggered grid to markers
        self.vym = interpolate(self.ctx.grid.xvy,       # x and y coordinates of the vy nodes
                               self.ctx.grid.yvy,
                               self.xm, self.ym,        # x and y coordinates of the markers
                               (self.ctx.model.vy,),    # tuple with the values to interpolate (vy)
                               indexing="equidistant")  # equidistant grid
                               
    def advect(self):
        # Advect markers
        self.xm += self.vxm * self.ctx.params.dt
        self.ym += self.vym * self.ctx.params.dt

        # Apply periodic boundary conditions
        self.xm = np.mod(self.xm, self.ctx.params.xsize)
        self.ym = np.mod(self.ym, self.ctx.params.ysize)

    def info(self):
        print(10*"-" + " Marker Pool Info " + 10*"-")
        print(f"Initialized {self.__class__.__name__} marker pool.")
        print(f"Number of markers in x-direction (per cell): {self.nmpcx:.1f}")
        print(f"Number of markers in y-direction (per cell): {self.nmpcy:.1f}")
        print(f"Number of markers per cell: {self.nmpcx*self.nmpcy:.1f}")
        print(f"Number of markers in x-direction: {self.nmx:.1f}")
        print(f"Number of markers in y-direction: {self.nmy:.1f}")
        print(f"Total number of markers: {self.nm:.1f}")
        print(f"Marker spacing in x-direction: {self.dxm:.1f}")
        print(f"Marker spacing in y-direction: {self.dym:.1f}")
        print(39*"-")

class RK42DStokes(Basic2DStokes):
    """
    Same as Basic2DStokes, but implements RK4 advection scheme.
    """
    def interpolate_vx(self, xm, ym):
        """
        Shortcut function to interpolate vx velocity from grid to markers.
        """
        return interpolate(self.ctx.grid.xvx,       # x and y coordinates of the vx nodes
                           self.ctx.grid.yvx, 
                           xm, ym,                  # x and y coordinates of the markers
                           (self.ctx.model.vx,),    # tuple with the values to interpolate (vx)
                           indexing="equidistant",  # equidistant grid
                           cont_corr="x")           # x-continuity correction  
    
    def interpolate_vy(self, xm, ym):
        """
        Shortcut function to interpolate vy velocity from grid to markers.
        """
         # Interpolate velocity from vy nodes of staggered grid to markers
        return interpolate(self.ctx.grid.xvy,       # x and y coordinates of the vy nodes
                           self.ctx.grid.yvy,
                           xm, ym,                  # x and y coordinates of the markers
                           (self.ctx.model.vy,),    # tuple with the values to interpolate (vy)
                           indexing="equidistant",  # equidistant grid
                           cont_corr="y")           # x-continuity correction

    def advect(self):
        #  ------- RK4 advection of markers -------
        #  Velocity in A = (x(m), y(m))
        xmA = self.xm
        ymA = self.ym
        vxmA = self.interpolate_vx(xmA, ymA)
        vymA = self.interpolate_vy(xmA, ymA)
        
        # Convenience variables
        dt = self.ctx.params.dt
        xsize = self.ctx.params.xsize
        ysize = self.ctx.params.ysize

        # Coordinates of B = (xA + 1/2 dt vxmA, yA + 1/2 dt vymA)
        xmB = np.mod(xmA + 1/2*dt*vxmA, xsize)
        ymB = np.mod(ymA + 1/2*dt*vymA, ysize)
        vxmB = self.interpolate_vx(xmB, ymB)
        vymB = self.interpolate_vy(xmB, ymB)

        # Coordinates of C = (xA + 1/2 dt vxmB, yA + 1/2 dt vymB)
        xmC = np.mod(xmA + 1/2*dt*vxmB, xsize)
        ymC = np.mod(ymA + 1/2*dt*vymB, ysize)
        vxmC = self.interpolate_vx(xmC, ymC)
        vymC = self.interpolate_vy(xmC, ymC)

        # Coordinates of D = (xA + dt vxmC, yA + dt vymC)
        xmD = np.mod(xmA + dt*vxmC, xsize)
        ymD = np.mod(ymA + dt*vymC, ysize)
        vxmD = self.interpolate_vx(xmD, ymD)
        vymD = self.interpolate_vy(xmD, ymD)

        vx_eff = (1/6)*(vxmA + 2*vxmB + 2*vxmC + vxmD)
        vy_eff = (1/6)*(vymA + 2*vymB + 2*vymC + vymD)

        # Update marker positions
        self.xm = np.mod(self.xm + dt*vx_eff, xsize)
        self.ym = np.mod(self.ym + dt*vy_eff, ysize)


        

       
