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
from Pyroclast.logging import get_logger

from Pyroclast.mpi import MPI, create_cart_comm

logger = get_logger(__name__)

class BasicStaggered2DMPI(BaseGrid):
    """
    Basic fully staggered grid for distributed 2D mic problems with MPI.
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

        # Assert MPI is enabled
        if MPI is None:
            raise RuntimeError("MPI module is not available. "
                               "BasicStaggered2DMPI requires MPI support.")
        
        # Set up MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        
        if not size > 1:
            logger.warning("BasicStaggered2DMPI initialized with a single MPI rank. "
                           "Consider using BasicStaggered2D for single-process runs.")
        
        # Set up grid spacing from global grid size
        s.dx = p.xsize_global / (p.nx_global - 1)
        s.dy = p.ysize_global / (p.ny_global - 1)
        
        # Store global physical domain bounds
        s.xmin_global = -s.dx
        s.xmax_global = p.xsize_global + s.dx
        s.ymin_global = -s.dy
        s.ymax_global = p.ysize_global + s.dy

        # Set up local grid size
        # We want to approximately match the 2D process grid to the aspect ratio
        # of the global grid discretization
        s.py, s.px = self.get_process_grid(size, p.ny_global, p.nx_global)

        # Setup cartesian communicator for 2D process grid
        dims = (s.py, s.px) # Domain decomposition in y and x
        periods = (False, False) # Non-periodic boundaries
        reorder = True # Allow rank reordering for efficiency
        
        # Create Cartesian communicator
        comm = create_cart_comm(dims=dims, periods=periods, reorder=reorder)
        rank = comm.Get_rank() # Rank from the Cartesian communicator, not COMM_WORLD

        # Compute local grid shape
        s.gi, s.gj = comm.Get_coords(rank) # Get process grid coordinates
                                       # gi = global i (y direction)
                                       # gj = global j (x direction)
        
        # Determine local grid shape
        s.ny, s.nx = self.get_local_grid_shape(p.ny_global,
                                               p.nx_global,
                                               s.py, s.px, s.gi, s.gj)


        # Determine global start indices for local grid
        s.istart, s.jstart = self.get_global_start_indices(p.ny_global,
                                                           p.nx_global,
                                                           s.py, s.px, s.gi, s.gj)
                
        # Compute bounds arrays for all processes (for I/O and ghost exchange)
        s.x_bounds = self.compute_bounds_array(p.nx_global, s.px, s.dx)
        s.y_bounds = self.compute_bounds_array(p.ny_global, s.py, s.dy)

        # Compute local domain size
        s.xmin, s.xmax = s.x_bounds[s.gj]
        s.ymin, s.ymax = s.y_bounds[s.gi]
        s.xsize = s.xmax - s.xmin
        s.ysize = s.ymax - s.ymin
    
        # Create local grid coordinates
        s.nx1 = s.nx + 2 + 1 # +2 for halo nodes, +1 for staggered grid ghost nodes
        s.ny1 = s.ny + 2 + 1

        # Create the main nodes
        s.x = np.linspace(s.xmin - s.dx, s.xmax + 2*s.dx, s.nx1)
        s.y = np.linspace(s.ymin - s.dy, s.ymax + 2*s.dy, s.ny1)

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
        self.info(ctx, comm)

    # Approximate optimal process grid based on global grid size
    # Idea: find Px, Py such that Px * Py = P and Px/Py ~ Nx/Ny 
    # Returns: (Px, Py)
    def get_process_grid(self, P, nx_g, ny_g):
        R = nx_g / ny_g
        best = None
        min_diff = float('inf')
        for px in range(1, P + 1):
            if P % px == 0:
                py = P // px
                diff = abs((px / py) - R)
                if diff < min_diff:
                    min_diff = diff
                    best = (py, px)
        return best  # (py, px)

    def get_local_grid_shape(self, ny_g, nx_g, py, px, gi, gj):
        # Split the grid among py * px processes in a 2D block decomposition
        nx_local = nx_g // px
        nx_remainder = nx_g % px
        ny_local = ny_g // py
        ny_remainder = ny_g % py

        # Distribute the remainder among the first few processes
        if gj < nx_remainder:
            nx_local += 1
        if gi < ny_remainder:
            ny_local += 1

        return ny_local, nx_local
    
    def get_global_start_indices(self, ny_g, nx_g, py, px, gi, gj):
        # Compute the starting global indices for the local grid
        start_i = (ny_g // py) * gi + min(gi, ny_g % py)
        start_j = (nx_g // px) * gj + min(gj, nx_g % px)
        return start_i, start_j
    
    def compute_bounds_array(self, n_global, n_procs, d, origin=0.0):
        bounds = np.empty((n_procs, 2), dtype=np.float64)

        # Compute number of cells per rank (same logic as in get_local_grid_shape)
        base = n_global // n_procs
        remainder = n_global % n_procs

        start_index = 0
        for i in range(n_procs):
            n_local = base + 1 if i < remainder else base
            end_index = start_index + n_local - 1
            bounds[i, 0] = origin + start_index * d
            bounds[i, 1] = origin + end_index * d
            start_index = end_index + 1

        return bounds

    def interpolate(self, ctx):
        """
        Interpolate density and viscosity from markers to grid nodes.
        """
        s, p, o = ctx

        rho = interpolate(s.xvy,                      # Density on y-velocity nodes
                          s.yvy,  
                          s.xm,                       # Marker x positions
                          s.ym,                       # Marker y positions
                          s.rhom,                     # Marker density
                          indexing="equidistant",     # Equidistant grid spacing
                          return_weights=False,       # Do not return weights
                          mpi=True)

        etab = interpolate(s.x,                       # Basic viscosity on grid nodes
                           s.y,
                           s.xm,                      # Marker x positions
                           s.ym,                      # Marker y positions
                           s.etam,                    # Marker viscosity
                           indexing="equidistant",    # Equidistant grid spacing
                           return_weights=False,      # Do not return weights
                           mpi=True)
        
        etap = interpolate(s.xp,                      # Pressure viscosity on grid nodes
                           s.yp,
                           s.xm,                      # Marker x positions
                           s.ym,                      # Marker y positions
                           s.etam,                    # Marker viscosity
                           indexing="equidistant",    # Equidistant grid spacing
                           return_weights=False,      # Do not return weights
                           mpi=True)

        mask = np.isfinite(rho)
        s.rho[mask] = rho[mask]

        mask = np.isfinite(etab)
        s.etab[mask] = etab[mask]

        mask = np.isfinite(etap)
        s.etap[mask] = etap[mask]

    def info(self, ctx, comm):
        s, p, o = ctx
        rank = comm.Get_rank()
        size = comm.Get_size()

        logger.info(10 * "-" + " Grid Information " + 10 * "-")
        logger.info("Basic staggered 2D grid initialized with MPI.")
        logger.info(f"Global domain size: {p.xsize_global:.3f} x {p.ysize_global:.3f}")
        logger.info(f"Global grid size: {p.nx_global} x {p.ny_global}")
        logger.info(f"Grid spacing: dx = {s.dx:.3f}, dy = {s.dy:.3f}")
        logger.info(f"Process grid shape: py x px = {s.py} x {s.px}")
        logger.info(f"Total number of MPI ranks: {size}")
        logger.info("")

        logger.info("Per-rank local grid layout:")
        header = (
            f"{'Rank':>4}  {'(gi,gj)':>8}  {'nx x ny':>9}  "
            f"{'[xmin, xmax]':>24}  {'[ymin, ymax]':>24}"
        )
        logger.info(header)
        logger.info("-" * len(header))

        for r in range(size):
            gi, gj = comm.Get_coords(r)
            istart, jstart = self.get_global_start_indices(
                p.ny_global, p.nx_global, s.py, s.px, gi, gj
            )
            ny_local, nx_local = self.get_local_grid_shape(
                p.ny_global, p.nx_global, s.py, s.px, gi, gj
            )

            xmin = jstart * s.dx
            xmax = xmin + (nx_local - 1) * s.dx
            ymin = istart * s.dy
            ymax = ymin + (ny_local - 1) * s.dy

            logger.info(
                f"{r:>4d}  ({gi:>2d},{gj:>2d})   "
                f"{nx_local:>4d}x{ny_local:<4d}   "
                f"[{xmin:>7.3f}, {xmax:>7.3f}]   "
                f"[{ymin:>7.3f}, {ymax:>7.3f}]"
            )

        logger.info("-" * len(header))



        


