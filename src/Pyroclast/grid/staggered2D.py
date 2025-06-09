"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: staggered2D.py
Description: This file implements 2D staggered grids with uniform spacing in each dimension.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np

from Pyroclast.grid.base_grid import BaseGrid
from Pyroclast.profiling import timer
from Pyroclast.interpolation.linear_2D_cpu \
    import interpolate_markers2grid as interpolate

class BasicStaggered2D(BaseGrid): # Inherit from BaseGrid
                                   # this automatically does some magic
    """
    Basic fully staggered grid for 2D mic problems.
    This class implements uniform grid spacing in each dimension.

    The class is suitable for simple 2D stokes problems.
    
    The system is assumed to have zero-origin at the top-left corner.
    
    The grid is staggered in the x and y directions and includes an extra
    row and column of ghost nodes.

    main nodes: eta_b (basic viscosity)
    x-velocity nodes: rho_vx
    y-velocity nodes: rho_vy
    pressure nodes: p, eta_p (viscosity on pressure nodes)
    """
    def __init__(self, ctx):
        """
        Initialization method for the grid.

        This method is only called once at the beginning of the simulation when
        starting from scratch. It is not called when restarting from a checkpoint and
        instead the state (all variables saved as attributes of "self") is restored.
        """

        # Read context
        s, p, o = ctx
        
        # All variables saved as attributes of "self"
        # will be available to all methods of the class.
        # Moreover, they will be automatically backed up
        # in simulation checkpoints.
        # Compute the grid spacing
        s.nx1 = p.nx + 1
        s.ny1 = p.ny + 1
        s.dx = p.xsize / (p.nx-1)
        s.dy = p.ysize / (p.ny-1)

        
        # Create the main nodes
        # Add an extra row and column of ghost nodes
        s.x = np.linspace(0, p.xsize + s.dx, s.nx1)
        s.y = np.linspace(0, p.ysize + s.dy, s.ny1)

        # Create the x-velocity nodes
        s.xvx = s.x
        s.yvx = s.y - s.dy/2

        # Create the y-velocity nodes
        s.xvy = s.x - s.dx/2
        s.yvy = s.y

        # Create the pressure nodes
        s.xp = s.x - s.dx/2
        s.yp = s.y - s.dy/2

        # Print some information about the grid
        self.info(ctx)
    
    def interpolate(self, ctx):
        """
        Interpolate density and viscosity from markers to grid nodes.
        """
        s, p, o = ctx

        rho = interpolate(s.xvy,                      # Density on y-velocity nodes
                            s.yvy,  
                            s.xm,                       # Marker x positions
                            s.ym,                       # Marker y positions
                            (s.rhom,),                  # Marker density
                            indexing="equidistant",     # Equidistant grid spacing
                            return_weights=False)       # Do not return weights

        
        etab = interpolate(s.x,                       # Basic viscosity on grid nodes
                             s.y,
                             s.xm,                      # Marker x positions
                             s.ym,                      # Marker y positions
                             (s.etam,),                 # Marker viscosity
                             indexing="equidistant",    # Equidistant grid spacing
                             return_weights=False)      # Do not return weights
        
        etap = interpolate(s.xp,                      # Pressure viscosity on grid nodes
                             s.yp,
                             s.xm,                      # Marker x positions
                             s.ym,                      # Marker y positions
                             (s.etam,),                 # Marker viscosity
                             indexing="equidistant",    # Equidistant grid spacing
                             return_weights=False)      # Do not return weights

        mask = np.isfinite(rho)
        s.rho[mask] = rho[mask]

        mask = np.isfinite(etab)
        s.etab[mask] = etab[mask]

        mask = np.isfinite(etap)
        s.etap[mask] = etap[mask]

    
    def info(self, ctx):
        s, p, o = ctx
        print(10*"-" + " Grid Information " + 10*"-")
        print("Basic staggered 2D grid initialized.")
        print(f"Domain size: {p.xsize:.1f} x {p.ysize:.1f}")
        print(f"Grid size: {s.nx1} x {s.ny1}")
        print(f"Grid spacing: {s.dx:.1f} x {s.dy:.1f}")
        print(f"Total nodes: {s.nx1 * s.ny1}")
        print(38*"-")
        


