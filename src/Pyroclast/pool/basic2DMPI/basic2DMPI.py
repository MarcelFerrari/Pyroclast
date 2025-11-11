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
from Pyroclast.linalg import clip_float as clip
import numba as nb
from Pyroclast.mpi import get_cart_comm, MPI

from .advection_routines import (
    N_DIRS,
    DIR_N, DIR_S, DIR_E, DIR_W,
    DIR_NE, DIR_NW, DIR_SE, DIR_SW,
    DIR_NO_MIGRATE,
    flag_migrating_markers,
    compact_markers,
    to_nb_container
)

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
        self.ctx = ctx
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
        s.nm = s.nmx * s.nmy  # Active number of markers
        
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
                            indexing="equidistant",  # equidistant grid
                            mpi=True)

        # Interpolate velocity from vy nodes of staggered grid to markers
        s.vym = interpolate(s.xvy,                   # x and y coordinates of the vy nodes
                            s.yvy,
                            s.xm, s.ym,              # x and y coordinates of the markers
                            s.vy,                    # values to interpolate (vy)
                            indexing="equidistant",  # equidistant grid
                            mpi=True
                            )
    
    # Allocate outbound marker buffers for MPI exchange
    def allocate_marker_buffers(self, nm):
        # 4 properties per marker: xm, ym, etam, rhom
        return tuple(np.zeros(nm, dtype=np.float64) for _ in range(4))

    # Return marker properties to advect
    def get_marker_properties(self):
        s, p, o = self.ctx
        return (s.xm, s.ym, s.etam, s.rhom)

    def update_references(self, ctx, nm_new, new_buffs):
        s, p, o = ctx
        s.nm = nm_new
        s.xm = new_buffs[0]
        s.ym = new_buffs[1]
        s.etam = new_buffs[2]
        s.rhom = new_buffs[3]

    def advect(self, ctx):
        s, p, o = ctx

        # Advect markers
        s.xm += s.vxm * s.dt
        s.ym += s.vym * s.dt

        # MPI setup
        comm = get_cart_comm()
        rank = comm.Get_rank()
        gi, gj = comm.Get_coords(rank)
        py, px = comm.Get_topo()[0]  # dims (rows, cols)

        # Get shifted rank coordinates
        def shift_coords(di, dj):
            i, j = gi + di, gj + dj
            if i < 0 or i >= py or j < 0 or j >= px:
                return MPI.PROC_NULL
            return comm.Get_cart_rank((i, j))

        # Clip physical boundaries
        clip(s.xm, s.xmin_global, s.xmax_global)
        clip(s.ym, s.ymin_global, s.ymax_global)

        # Count migrating markers AFTER clipping
        still_m_count, out_m_counts = flag_migrating_markers(
            s.nm, s.xm, s.ym, s.m_xmin, s.m_xmax, s.m_ymin, s.m_ymax, nb.get_num_threads()
        )

        # Prepare to receive incoming marker counts
        in_m_counts = np.zeros_like(out_m_counts)

        # Define neighbor ranks
        directions = [
            shift_coords(-1, 0),   # N
            shift_coords(1, 0),    # S
            shift_coords(0, 1),    # E
            shift_coords(0, -1),   # W
            shift_coords(-1, 1),   # NE
            shift_coords(-1, -1),  # NW
            shift_coords(1, 1),    # SE
            shift_coords(1, -1),   # SW
        ]

        # Exchange migration counts
        reqs = []
        for i, nbr_rank in enumerate(directions):
            if nbr_rank != MPI.PROC_NULL:
                # Use slices to keep the type ndarrays
                reqs.append(comm.Irecv(in_m_counts[i:i+1], source=nbr_rank))
                reqs.append(comm.Isend(out_m_counts[i:i+1], dest=nbr_rank))
                

        MPI.Request.Waitall(reqs)
        
        # Compute new total marker count after exchange
        nm_new = s.nm - np.sum(out_m_counts) + np.sum(in_m_counts)
        assert nm_new == still_m_count + np.sum(in_m_counts), \
            f"Marker count mismatch after exchange: {nm_new} != {still_m_count} + {np.sum(in_m_counts)}"
        
        # Prepare outbound marker data & reallocate arrays for inbound markers
        # Read marker properties via function to allow overriding in child classes
        marker_properties = self.get_marker_properties()
        new_marker_properties = self.allocate_marker_buffers(nm_new)

        # Compute offsets for incoming markers
        # We start writing after the still markers, which have been
        # moved to the beginning of the arrays during compaction
        # 1. Prepare inbound offsets
        inbound_write_offsets = np.zeros(N_DIRS + 1, dtype=np.int64)
        inbound_write_offsets[1:] = np.cumsum(in_m_counts)
        inbound_write_offsets += still_m_count
        
        assert inbound_write_offsets[-1] == nm_new, \
            f"Write offsets do not match new marker count: {inbound_write_offsets[-1]} != {nm_new}"
        assert inbound_write_offsets[0] == still_m_count, \
            f"First write offset does not match still marker count: {inbound_write_offsets[0]} != {still_m_count}"

        reqs = []  # to accumulate MPI requests
        out_buffs_list = []  # to hold outbound buffers per property

        # 2. Iterate property by property
        for tag, (prop, new_prop) in enumerate(zip(marker_properties, new_marker_properties)):
            # out_buffs = (out_N, out_S, out_E, out_W, out_NE, out_NW, out_SE, out_SW)
            
            # Allocate outbound buffers
            out_buffs = tuple(
                np.empty(out_m_counts[dir_idx], dtype=np.float64)
                for dir_idx in range(N_DIRS)
            )
            out_buffs_list.append(out_buffs) # Need to keep them alive for MPI

            compact_markers(
                s.nm,
                s.xm, s.ym,
                s.m_xmin, s.m_xmax, s.m_ymin, s.m_ymax,
                prop,
                *out_buffs,  # unpack all 8 arrays
                new_prop,    # array for still markers
                nb.get_num_threads()
            )

            # Start MPI communication for this property immediately
            for dir_idx, nbr_rank in enumerate(directions):
                if nbr_rank == MPI.PROC_NULL:
                    continue

                # Non-blocking receive
                if in_m_counts[dir_idx] > 0:
                    start, end = inbound_write_offsets[dir_idx], inbound_write_offsets[dir_idx + 1]
                    reqs.append(comm.Irecv(new_prop[start:end], source=nbr_rank, tag=tag))

                # Non-blocking send
                if out_m_counts[dir_idx] > 0:
                    reqs.append(comm.Isend(out_buffs[dir_idx], dest=nbr_rank, tag=tag))


        MPI.Request.Waitall(reqs)

        # Update active marker count
        self.update_references(ctx, nm_new, new_marker_properties)

    def info(self, ctx):
        s, p, o = ctx
        logger.info(10 * "-" + " Marker Pool Info " + 10 * "-")
        logger.info(f"Initialized {self.__class__.__name__} marker pool.")
        logger.info(f"Markers per cell: {p.nmpcx * p.nmpcy}")
        logger.info(f"Markers per domain: {s.nm}")
        logger.info(f"dxm: {s.dxm:.2f}, dym: {s.dym:.2f}")
        logger.info(39 * "-")