"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: basic2D.py
Description: This file implements 2D marker pools for simple inclusion problems.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np
import numba as nb

from Pyroclast.pool.basic2D import Basic2DStokes


class CircularInclusion(Basic2DStokes):
    """
    Circular inclusion in 2D.
    """
    def __init__(self, ctx):
        # Initialize markers
        super().__init__(ctx)

        # Read context
        s, p, o = ctx

        # Define parameters for a circular inclusion
        x0 = p.xsize/2. # x coordinate of the center of the circle
        y0 = p.ysize/2. # y coordinate of the center of the circle
        r = p.r # radius of the circle
        

        # Marker material properties
        s.rhom = np.zeros(s.nm, dtype=np.float64)
        s.etam = np.zeros(s.nm, dtype=np.float64)


        s.rhom, s.etam = _init_circular_inclusion(s.nmx, s.nmy,
                                                  s.dxm, s.dym,
                                                  x0, y0, r,
                                                  s.xm, s.ym,
                                                  p.rho_plume,
                                                  p.rho_mantle,
                                                  p.eta_plume,
                                                  p.eta_mantle,
                                                  s.rhom, s.etam)

@nb.njit(cache=True)
def _init_circular_inclusion(nmx, nmy, dxm, dym, x0, y0, r, xm, ym, rho_plume, rho_mantle, eta_plume, eta_mantle, rhom, etam):
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
            xm[m] = dxm/2 + j*dxm + (np.random.uniform()-0.5)*dxm
            ym[m] = dym/2 + i*dym + (np.random.uniform()-0.5)*dym

            # Set up material properties of markers
            d = (xm[m] - x0)**2 + (ym[m] - y0)**2

            if d < r**2:
                rhom[m] = rho_plume
                etam[m] = eta_plume
            else:
                rhom[m] = rho_mantle
                etam[m] = eta_mantle

            m += 1
    return rhom, etam

# class SquareInclusion(Basic2DStokes):
#     """
#     Square inclusion in 2D.
#     """
#     def initialize(self):
#         # Initialize markers
#         super().initialize()

#         # Define parameters for a circular inclusion
#         self.x0 = self.ctx.params.xsize/2. # x coordinate of the center of the circle
#         self.y0 = self.ctx.params.ysize/2. # y coordinate of the center of the circle
#         self.r = self.ctx.params.r # radius of the circle
        
#         # Marker material properties
#         self.eta_plume = self.ctx.params.eta_plume
#         self.eta_mantle = self.ctx.params.eta_mantle
#         self.rho_plume = self.ctx.params.rho_plume 
#         self.rho_mantle = self.ctx.params.rho_mantle

#         # Marker material properties
#         self.rhom = np.zeros(self.nm, dtype=np.float64)
#         self.etam = np.zeros(self.nm, dtype=np.float64)


#         self._init_markers(self.nmx, self.nmy,
#                     self.dxm, self.dym,
#                     self.x0, self.y0, self.r,
#                     self.xm, self.ym,
#                     self.rhom, self.etam,
#                     self.eta_plume,
#                     self.eta_mantle,
#                     self.rho_plume,
#                     self.rho_mantle)

#     @staticmethod
#     @nb.njit(cache=True)
#     def _init_markers(nmx, nmy, dxm, dym, x0, y0, r, xm, ym, rhom, etam, eta_plume, eta_mantle, rho_plume, rho_mantle):
#         """
#         Initialize the material properties of the markers.

#         nmx: Number of markers in x direction
#         nmy: Number of markers in y direction
#         dxm: Marker spacing in x direction
#         dym: Marker spacing in y direction
#         x0: x coordinate of the center of the circle
#         y0: y coordinate of the center of the circle
#         r: radius of the circle
#         xm: Marker positions in x direction
#         ym: Marker positions in y direction
#         rhom: Marker density
#         etam: Marker viscosity
#         """
#         m = 0
#         # Initialize marker values
#         for i in range(nmy):
#             for j in range(nmx):
#                 # Compute marker index
#                 xm[m] = dxm/2 + j*dxm + (np.random.uniform()-0.5)*dxm
#                 ym[m] = dym/2 + i*dym + (np.random.uniform()-0.5)*dym

#                 # Set up square inclusion
#                 if xm[m] > x0 - r and xm[m] < x0 + r and ym[m] > y0 - r and ym[m] < y0 + r:
#                     rhom[m] = rho_plume
#                     etam[m] = eta_plume
#                 else:
#                     rhom[m] = rho_mantle
#                     etam[m] = eta_mantle

#                 m += 1

# class FreeSurfaceInclusion(RK42DStokes):
#     """
#     We are solving the Stokes equation in 2D using the 
#     Marker-in-cell method for a circular inclusion in 2D.
    
#     Basic2DStokes implements basic marker advection and
#     interpolation for the Stokes equation in 2D.
#     The marker quantities are density rhom and
#     viscosity etam. The grid x and y velocities are
#     interpolated to the markers for the advection step.

#     The model is defined as follows:
#     - Inside the circle: rhom = 3200.0, etam = 1e18
#     - Outside the circle: rhom = 3300.0, etam = 1e19
#     """
#     def initialize(self):
#         # Initialize markers
#         super().initialize()

#         # Define parameters for a circular inclusion
#         self.x0 = self.ctx.params.xsize/2. # x coordinate of the center of the circle
#         self.y0 = self.ctx.params.ysize/2. # y coordinate of the center of the circle
#         self.r = self.ctx.params.r # radius of the circle

#         # Marker material properties
#         self.rhom = np.zeros(self.nm, dtype=np.float64)
#         self.etam = np.zeros(self.nm, dtype=np.float64)

#         self._init_markers(self.nmx, self.nmy,
#                     self.dxm, self.dym,
#                     self.x0, self.y0, self.r,
#                     self.xm, self.ym,
#                     self.rhom, self.etam)

#     @staticmethod
#     @nb.njit(cache=True)
#     def _init_markers(nmx, nmy, dxm, dym, x0, y0, r, xm, ym, rhom, etam):
#         """
#         Initialize the material properties of the markers.

#         nmx: Number of markers in x direction
#         nmy: Number of markers in y direction
#         dxm: Marker spacing in x direction
#         dym: Marker spacing in y direction
#         x0: x coordinate of the center of the circle
#         y0: y coordinate of the center of the circle
#         r: radius of the circle
#         xm: Marker positions in x direction
#         ym: Marker positions in y direction
#         rhom: Marker density
#         etam: Marker viscosity
#         """

#         m = 0
#         # Initialize marker values
#         for i in range(nmy):
#             for j in range(nmx):
#                 # Compute marker index
#                 xm[m] = dxm/2 + (j-1)*dxm + (np.random.uniform()-0.5)*dxm
#                 ym[m] = dym/2 + (i-1)*dym + (np.random.uniform()-0.5)*dym

#                 # Set up material properties of markers
#                 d = (xm[m] - x0)**2 + (ym[m] - y0)**2

#                 if d < r**2:
#                     rhom[m] = 3200.0
#                     etam[m] = 1e18
#                 else:
#                     rhom[m] = 3300.0
#                     etam[m] = 1e19

#                 # Air layer
#                 if ym[m] < 20000:
#                     rhom[m] = 1.0
#                     etam[m] = 1e16

#                 m += 1
   
# class EllipseInclusion(RK42DStokes):
#     """
#     We are solving the Stokes equation in 2D using the 
#     Marker-in-cell method for a circular inclusion in 2D.
    
#     Basic2DStokes implements basic marker advection and
#     interpolation for the Stokes equation in 2D.
#     The marker quantities are density rhom and
#     viscosity etam. The grid x and y velocities are
#     interpolated to the markers for the advection step.

#     The model is defined as follows:
#     - Inside the circle: rhom = 3200.0, etam = 1e18
#     - Outside the circle: rhom = 3300.0, etam = 1e19
#     """
#     def initialize(self):
#         # Define parameters for a circular inclusion
#         self.x0 = self.ctx.params.xsize/2. # x coordinate of the center of the circle
#         self.y0 = self.ctx.params.ysize/2. # y coordinate of the center of the circle
#         # self.r = self.ctx.params.r # radius of the circle
#         self.a = self.ctx.params.a
#         self.b = self.ctx.params.b

#         # Initialize markers
#         super().initialize()


#     def init_markers(self):
#         self._init_markers(self.nmx, self.nmy,
#                            self.dxm, self.dym,
#                            self.x0, self.y0,
#                             self.a, self.b,
#                            self.xm, self.ym,
#                            self.rhom, self.etam)

#     @staticmethod
#     @nb.njit(cache=True)
#     def _init_markers(nmx, nmy, dxm, dym, x0, y0, a, b, xm, ym, rhom, etam):
#         """
#         Initialize the material properties of the markers.

#         nmx: Number of markers in x direction
#         nmy: Number of markers in y direction
#         dxm: Marker spacing in x direction
#         dym: Marker spacing in y direction
#         x0: x coordinate of the center of the circle
#         y0: y coordinate of the center of the circle
#         r: radius of the circle
#         xm: Marker positions in x direction
#         ym: Marker positions in y direction
#         rhom: Marker density
#         etam: Marker viscosity
#         """

#         m = 0
#         # Initialize marker values
#         for i in range(nmy):
#             for j in range(nmx):
#                 # Compute marker index
#                 xm[m] = dxm/2 + (j-1)*dxm + (np.random.uniform()-0.5)*dxm
#                 ym[m] = dym/2 + (i-1)*dym + (np.random.uniform()-0.5)*dym

#                 # Set up material properties of markers
#                 d = ((xm[m] - x0)/a)**2 + ((ym[m] - y0)/b)**2

#                 if d < 1.0:
#                     rhom[m] = 3200.0
#                     etam[m] = 1e18
#                 else:
#                     rhom[m] = 3300.0
#                     etam[m] = 1e19

#                 m += 1