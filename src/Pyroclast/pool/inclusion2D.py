"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: basic2D.py
Description: This file implements 2D marker pools for simple inclusion problems.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

See LICENSE file in the project root for full license information.
"""

import numpy as np
import numba as nb

from Pyroclast.pool.basic2D import Basic2DStokes


class CircularInclusion(Basic2DStokes):
    """
    We are solving the Stokes equation in 2D using the 
    Marker-and-Cell method for a circular inclusion in 2D.
    
    Basic2DStokes implements basic marker advection and
    interpolation for the Stokes equation in 2D.
    The marker quantities are density rhom and
    viscosity etam. The grid x and y velocities are
    interpolated to the markers for the advection step.

    The model is defined as follows:
    - Inside the circle: rhom = 3200.0, etam = 1e18
    - Outside the circle: rhom = 3300.0, etam = 1e19
    """
    def initialize(self):
        # Define parameters for a circular inclusion
        self.x0 = self.ctx.params.xsize/2. # x coordinate of the center of the circle
        self.y0 = self.ctx.params.ysize/2. # y coordinate of the center of the circle
        self.r = self.ctx.params.r # radius of the circle

        # Initialize markers
        super().initialize()

        # Dump initial markers as npz file
        #np.savez("initial_markers.npz", xm=self.xm, ym=self.ym, rhom=self.rhom, etam=self.etam)
        #exit(0)

    def init_markers(self):
        self._init_markers(self.nmx, self.nmy,
                           self.dxm, self.dym,
                           self.x0, self.y0, self.r,
                           self.xm, self.ym,
                           self.rhom, self.etam)

    @staticmethod
    @nb.njit(cache=True)
    def _init_markers(nmx, nmy, dxm, dym, x0, y0, r, xm, ym, rhom, etam):
        """
        Initialize the material properties of the markers.

        nmx: Number of markers in x direction
        nmy: Number of markers in y direction
        dxm: Marker spacing in x direction
        dym: Marker spacing in y direction
        x0: x coordinate of the center of the circle
        y0: y coordinate of the center of the circle
        r: radius of the circle
        xm: Marker positions in x direction
        ym: Marker positions in y direction
        rhom: Marker density
        etam: Marker viscosity
        """

        m = 0
        # Initialize marker values
        for i in range(nmy):
            for j in range(nmx):
                # Compute marker index
                xm[m] = dxm/2 + (j-1)*dxm + (np.random.uniform()-0.5)*dxm
                ym[m] = dym/2 + (i-1)*dym + (np.random.uniform()-0.5)*dym

                # Set up material properties of markers
                d = (xm[m] - x0)**2 + (ym[m] - y0)**2

                if d < r**2:
                    rhom[m] = 3200.0
                    etam[m] = 1e18
                else:
                    rhom[m] = 3300.0
                    etam[m] = 1e19

                m += 1

class EllipseInclusion(Basic2DStokes):
    """
    We are solving the Stokes equation in 2D using the 
    Marker-and-Cell method for a circular inclusion in 2D.
    
    Basic2DStokes implements basic marker advection and
    interpolation for the Stokes equation in 2D.
    The marker quantities are density rhom and
    viscosity etam. The grid x and y velocities are
    interpolated to the markers for the advection step.

    The model is defined as follows:
    - Inside the circle: rhom = 3200.0, etam = 1e18
    - Outside the circle: rhom = 3300.0, etam = 1e19
    """
    def initialize(self):
        # Define parameters for a circular inclusion
        self.x0 = self.ctx.params.xsize/2. # x coordinate of the center of the circle
        self.y0 = self.ctx.params.ysize/2. # y coordinate of the center of the circle
        # self.r = self.ctx.params.r # radius of the circle
        self.a = self.ctx.params.a
        self.b = self.ctx.params.b

        # Initialize markers
        super().initialize()

        # Dump initial markers as npz file
        #np.savez("initial_markers.npz", xm=self.xm, ym=self.ym, rhom=self.rhom, etam=self.etam)
        #exit(0)

    def init_markers(self):
        self._init_markers(self.nmx, self.nmy,
                           self.dxm, self.dym,
                           self.x0, self.y0,
                            self.a, self.b,
                           self.xm, self.ym,
                           self.rhom, self.etam)

    @staticmethod
    @nb.njit(cache=True)
    def _init_markers(nmx, nmy, dxm, dym, x0, y0, a, b, xm, ym, rhom, etam):
        """
        Initialize the material properties of the markers.

        nmx: Number of markers in x direction
        nmy: Number of markers in y direction
        dxm: Marker spacing in x direction
        dym: Marker spacing in y direction
        x0: x coordinate of the center of the circle
        y0: y coordinate of the center of the circle
        r: radius of the circle
        xm: Marker positions in x direction
        ym: Marker positions in y direction
        rhom: Marker density
        etam: Marker viscosity
        """

        m = 0
        # Initialize marker values
        for i in range(nmy):
            for j in range(nmx):
                # Compute marker index
                xm[m] = dxm/2 + (j-1)*dxm + (np.random.uniform()-0.5)*dxm
                ym[m] = dym/2 + (i-1)*dym + (np.random.uniform()-0.5)*dym

                # Set up material properties of markers
                d = ((xm[m] - x0)/a)**2 + ((ym[m] - y0)/b)**2

                if d < 1.0:
                    rhom[m] = 3200.0
                    etam[m] = 1e18
                else:
                    rhom[m] = 3300.0
                    etam[m] = 1e19

                m += 1