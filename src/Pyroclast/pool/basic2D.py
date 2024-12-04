"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: basic2D.py
Description: This file implements basic marker pool operations for 2D staggered grids.
             with uniform spacing in each dimension.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

See LICENSE file in the project root for full license information.
"""

import numpy as np
import numba as nb
from random import uniform

from Pyroclast.pool.base_pool import BasePool
from Pyroclast.interpolation.linear_2D_cpu \
    import interpolate_grid2markers as interpolate

class Basic2DPool(BasePool): # Inherit from BasePool
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
    
    def advect(self):
        # Advect markers
        self.xm += self.vxm * self.ctx.params.dt
        self.ym += self.vym * self.ctx.params.dt

        # Apply periodic boundary conditions
        self.xm = np.mod(self.xm, self.ctx.params.xsize)
        self.ym = np.mod(self.ym, self.ctx.params.ysize)

class Basic2DStokes(Basic2DPool):
    """
    Base class for 2D stokes flow.
    """
    def initialize(self):
        # Initialize pool coordinates and velocities
        super().initialize()

        # Marker material properties
        self.rhom = np.zeros(self.nm, dtype=np.float64)
        self.etam = np.zeros(self.nm, dtype=np.float64)

        self.init_markers()

        # Print some information about the pool
        self.info()

    def init_markers(self):
        """
        Initialize the material properties of the markers.
        This will be implemented for the specific problems
        """
        raise NotImplementedError("init_markers method must be implemented in derived class.")
    
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
                               indexing="equidistant",  # equidistant grid
                               return_weights=False)    # do not return interpolation weights
        
        # Interpolate velocity from vy nodes of staggered grid to markers
        self.vym = interpolate(self.ctx.grid.xvy,       # x and y coordinates of the vy nodes
                               self.ctx.grid.yvy,
                               self.xm, self.ym,        # x and y coordinates of the markers
                               (self.ctx.model.vy,),    # tuple with the values to interpolate (vy)
                               indexing="equidistant",  # equidistant grid
                               return_weights=False)    # do not return interpolation weights
        
    def info(self):
        print(10*"-" + " Marker Pool Info " + 10*"-")
        print(f"Initialized {self.__class__.__name__} marker pool.")
        print(f"Number of markers in x-direction (per cell): {self.nmpcx}")
        print(f"Number of markers in y-direction (per cell): {self.nmpcy}")
        print(f"Number of markers per cell: {self.nmpcx*self.nmpcy}")
        print(f"Number of markers in x-direction: {self.nmx}")
        print(f"Number of markers in y-direction: {self.nmy}")
        print(f"Total number of markers: {self.nm}")
        print(f"Marker spacing in x-direction: {self.dxm}")
        print(f"Marker spacing in y-direction: {self.dym}")
        print(39*"-")


        

       
