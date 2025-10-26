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
from Pyroclast.logging import get_logger
from Pyroclast.utils import wrap_periodic
import numba as nb

logger = get_logger(__name__)

class Basic2DStokesMPI(BasePool): # Inherit from BasePool
                               # this automatically does some magic
    """
        This class implements a basic pool of markers for 2D staggered grids.
        This is not meant to be used directly, but to be inherited by specific
        pool implementations.
    """
    def __init__(self, ctx):
        # The marker pool is always initialized after the grid
        # this means that any necessary grid information is already
        # available via the context.

        # Read context
        s, p, o = ctx
        
        # Each patch is responsible for markers in 
        # its own extended domain [xmin - dx, xmax] x [ymin - dy, ymax]
        # This includes the left/top no-mans land between patches or boundary values.
        # Ranks interfacing to the bottom or right physical boundary will also need
        # to extend to ymax + dy or xmax + dx to handle markers over boundary values.
        s.m_xmin = s.xmin - s.dx
        s.m_xmax = s.xmax
        s.m_ymin = s.ymin - s.dy
        s.m_ymax = s.ymax

        # Compute local number of markers
        # in each cell, including top/left no-mans land
        s.nmx = s.nx * p.nmpcx
        s.nmy = s.ny * p.nmpcy

        # Check if we are top/left boundary ranks
        if s.gi == s.py - 1:  # Bottom boundary rank
            s.nmy += p.nmpcy  # Add extra row of markers
            s.m_ymax += s.dy  # Extend marker domain upwards
        
        if s.gj == s.px - 1:  # Right boundary rank
            s.nmx += p.nmpcx  # Add extra column of markers
            s.m_xmax += s.dx  # Extend marker domain rightwards

        # Total number of markers
        s.nm = s.nmx * s.nmy

        # Compute marker spacing
        s.dxm = (s.m_xmax - s.m_xmin) / s.nmx
        s.dym = (s.m_ymax - s.m_ymin) / s.nmy

        # Init marker arrays
        # Marker positions
        s.xm = np.zeros(s.nm, dtype=np.float64)
        s.ym = np.zeros(s.nm, dtype=np.float64)
        
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
                            s.vx,                    # values to interpolate (vx)
                            indexing="equidistant")  # equidistant grid
        
        # Interpolate velocity from vy nodes of staggered grid to markers
        s.vym = interpolate(s.xvy,                   # x and y coordinates of the vy nodes
                            s.yvy,
                            s.xm, s.ym,              # x and y coordinates of the markers
                            s.vy,                    # values to interpolate (vy)
                            indexing="equidistant")  # equidistant grid
                               
    def advect(self, ctx):
        s, p, o = ctx

        # Advect markers
        s.xm += s.vxm * s.dt
        s.ym += s.vym * s.dt

        # Apply periodic boundary conditions
        # We use distributed interpolation to propagate information
        # across periodic boundaries
        s.xm = wrap_periodic(s.xm, s.m_xmin, s.m_xmax)
        s.ym = wrap_periodic(s.ym, s.m_ymin, s.m_ymax)
        

    def info(self, ctx):
        s, p, o = ctx

        logger.info(10*"-" + " Marker Pool Info " + 10*"-")
        logger.info(f"Initialized {self.__class__.__name__} marker pool.")
        logger.info(f"Number of markers in x-direction (per cell): {p.nmpcx}")
        logger.info(f"Number of markers in y-direction (per cell): {p.nmpcy}")
        logger.info(f"Number of markers per cell: {p.nmpcx*p.nmpcy}")
        logger.info(f"Number of markers in x-direction: {s.nmx}")
        logger.info(f"Number of markers in y-direction: {s.nmy}")
        logger.info(f"Total number of markers per domain: {s.nm}")
        logger.info(f"Marker spacing in x-direction: {s.dxm:.1f}")
        logger.info(f"Marker spacing in y-direction: {s.dym:.1f}")
        logger.info(39*"-")



class RectangularInclusionMPI(Basic2DStokesMPI):
    """
    Circular inclusion in 2D.
    """
    def __init__(self, ctx):
        # Initialize markers
        super().__init__(ctx)

        # Read context
        s, p, o = ctx

        # Define parameters for a circular inclusion
        x0 = p.xsize_global/2. # x coordinate of the center of the rectangle
        y0 = p.ysize_global/2. # y coordinate of the center of the rectangle
        h = p.h # height of the rectangle
        w = p.w # width of the rectangle
        
        # Marker material properties
        s.rhom = np.zeros(s.nm, dtype=np.float64)
        s.etam = np.zeros(s.nm, dtype=np.float64)


        s.rhom, s.etam = _init_rectangular_inclusion(s.nmx, s.nmy,
                                                     s.m_xmin, s.m_ymin,
                                                     s.dxm, s.dym,
                                                     x0, y0, w, h,
                                                     s.xm, s.ym,
                                                     p.rho_plume,
                                                     p.rho_mantle,
                                                     p.eta_plume,
                                                     p.eta_mantle,
                                                     s.rhom, s.etam)
        
        # Apply periodic boundary conditions to marker positions
        s.xm = wrap_periodic(s.xm, s.m_xmin, s.m_xmax)
        s.ym = wrap_periodic(s.ym, s.m_ymin, s.m_ymax)

        
@nb.njit(cache=True)
def _init_rectangular_inclusion(nmx, nmy, xmin, ymin, dxm, dym, x0, y0, w, h, xm, ym, rho_plume, rho_mantle, eta_plume, eta_mantle, rhom, etam):
    """
    Initialize the material properties of the markers.

    nmx: Number of markers in x direction
    nmy: Number of markers in y direction
    dxm: Marker spacing in x direction
    dym: Marker spacing in y direction
    x0: x coordinate of the center of the rectangle
    y0: y coordinate of the center of the rectangle
    w: width of the rectangle
    h: height of the rectangle
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
            xm[m] = dxm/2 + j*dxm + (np.random.uniform()-0.5)*dxm + xmin
            ym[m] = dym/2 + i*dym + (np.random.uniform()-0.5)*dym + ymin

            # Set up material properties of markers
            if (xm[m] > x0 - w/2 and xm[m] < x0 + w/2 and
                ym[m] > y0 - h/2 and ym[m] < y0 + h/2):
                rhom[m] = rho_plume
                etam[m] = eta_plume
            else:
                rhom[m] = rho_mantle
                etam[m] = eta_mantle

            m += 1
    return rhom, etam