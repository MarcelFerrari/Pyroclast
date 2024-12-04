"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: staggered2D.py
Description: This file implements 2D staggered grids with uniform spacing in each dimension.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

See LICENSE file in the project root for full license information.
"""

import numpy as np

from Pyroclast.grid.base_grid import BaseGrid

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
    def initialize(self):
        """
        Initialization method for the grid.

        This method is only called once at the beginning of the simulation when
        starting from scratch. It is not called when restarting from a checkpoint and
        instead the state (all variables saved as attributes of "self") is restored.
        """

        # Read constants from the context
        # These parameters are set in the input file
        # and are stored in the "params" object of the context.
        # Whatever variable you pass under the [params] annotation
        # in the input file will be available here as an attribute.
        self.xsize = self.ctx.params.xsize
        self.ysize = self.ctx.params.ysize
        self.nx = self.ctx.params.nx
        self.ny = self.ctx.params.ny

        # All variables saved as attributes of "self"
        # will be available to all methods of the class.
        # Moreover, they will be automatically backed up
        # in simulation checkpoints.
        # Compute the grid spacing
        self.nx1 = self.nx + 1
        self.ny1 = self.ny + 1
        self.dx = self.xsize / (self.nx-1)
        self.dy = self.ysize / (self.ny-1)

        # You can define arbitrary helper functions to
        # keep the code clean and organized
        self.create_grid()

        # Print some information about the grid
        self.info()

    def create_grid(self):
        # Create the main nodes
        # Add an extra row and column of ghost nodes
        self.x = np.linspace(0, self.xsize + self.dx, self.nx+1)
        self.y = np.linspace(0, self.ysize + self.dy, self.ny+1)

        # Create the x-velocity nodes
        self.xvx = self.x
        self.yvx = self.y - self.dy/2

        # Create the y-velocity nodes
        self.xvy = self.x - self.dx/2
        self.yvy = self.y

        # Create the pressure nodes
        self.xp = self.x - self.dx/2
        self.yp = self.y - self.dy/2

        # Create additional nodes
        # It is fine to copy additional arrays for clarity as they are not
        # actually stored in memory, but are just references to the same data.
        # Density nodes
        self.xrho_vx = self.xvx
        self.yrho_vx = self.yvx

        self.xrho_vy = self.xvy
        self.yrho_vy = self.yvy
        
        # Basic viscosity nodes
        self.xeta_b = self.x
        self.yeta_b = self.y

        # Pressure viscosity nodes
        self.xeta_p = self.xp
        self.yeta_p = self.yp
        
    def info(self):
        print(10*"-" + " Grid Information " + 10*"-")
        print("Basic staggered 2D grid initialized.")
        print(f"Domain size: {self.xsize} x {self.ysize}")
        print(f"Grid size: {self.nx} x {self.ny}")
        print(f"Grid spacing: {self.dx} x {self.dy}")
        print(f"Total nodes: {self.nx1 * self.ny1}")
        print(38*"-")
        


