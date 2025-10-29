"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: advection.py
Description: this file contains the implementation of a simple
             constant velocity advection model in 2D.
             This is meant to test marker advection.
             

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
import numba as nb
import numpy as np
from Pyroclast.model.base_model import BaseModel
from Pyroclast.profiling import timer
from Pyroclast.logging import get_logger
from Pyroclast.mpi import get_cart_comm, MPI, get_shard_filename

logger = get_logger(__name__)


# Model class
class ConstantVelocityAdvection2DMPI(BaseModel):
    """
    Constant velocity advection model in 2D.
     
    """
    def __init__(self, ctx):
        # At this point the grid and markers are already initialized
        s, p, o = ctx

        # Extra options (with default values if not set in the input file)
        self.BC = p.get('BC', -1)      # Boundary condition parameter
        self.gy = p.get('gy', 10.0)    # Gravity constant

        # Set up rho, eta_b, and eta_p arrays
        # These are not defined on ghost nodes
        s.rho = np.zeros((s.ny1, s.nx1))
        s.etab = np.zeros((s.ny1, s.nx1))
        s.etap = np.zeros((s.ny1, s.nx1))
        
        # Set border values to nan
        s.rho[-1, :] = np.nan
        s.rho[:, -1] = np.nan
        s.etab[-1, :] = np.nan
        s.etab[:, -1] = np.nan
        s.etap[-1, :] = np.nan
        s.etap[:, -1] = np.nan
        
        # Init vx and vy arrays
        s.vx = np.zeros((s.ny1, s.nx1))
        s.vy = np.zeros((s.ny1, s.nx1))
        
        # Initialize velocity field
        init_velocity_field(s.yvx, s.vx, s.xvy, s.vy, \
                             p.xsize_global, p.ysize_global)


        # Set up time step
        vxmax_local = np.array([np.max(np.abs(s.vx))])
        vymax_local = np.array([np.max(np.abs(s.vy))])

        comm = get_cart_comm()

        vxmax_global = np.zeros_like(vxmax_local)
        vymax_global = np.zeros_like(vymax_local)
        comm.Allreduce(vxmax_local, vxmax_global, op=MPI.MAX)
        comm.Allreduce(vymax_local, vymax_global, op=MPI.MAX)

        s.vxmax_global = vxmax_global[0]
        s.vymax_global = vymax_global[0]

        vxmax = max(vxmax_global[0], 1e-12)
        vymax = max(vymax_global[0], 1e-12)

        # Compute time step
        dty_phys = p.cfl_dispmax * p.L / vymax
        dtx_phys = p.cfl_dispmax * p.L / vxmax
        dtx_halo = p.cfl_dispmax * s.dx / vxmax
        dty_halo = p.cfl_dispmax * s.dy / vymax

        dt_local = np.array([min(dtx_phys, dty_phys, dtx_halo, dty_halo)])
        dt_global = np.zeros_like(dt_local)
        comm.Allreduce(dt_local, dt_global, op=MPI.MIN)

        self.dt = dt_global[0]

        assert s.vx.shape == (s.ny1, s.nx1), "vx shape mismatch"
        assert s.vy.shape == (s.ny1, s.nx1), "vy shape mismatch"

        self.frame = 0
        self.zpad = len(str(p.max_iterations//o.framedump_interval)) + 1

    def update_time_step(self, ctx):
        # Read the context
        s, p, o = ctx
        s.dt = self.dt

    def dump(self, ctx):
        # Read the context
        s, p, o = ctx

        # Dump state to file
        fname = f"frame_{str(self.frame).zfill(self.zpad)}"
        fname = get_shard_filename(fname) + f".npz"

        np.savez(fname, vx=s.vx, vy=s.vy,
                 rho=s.rho, etab=s.etab, etap=s.etap,
                 coords=(s.gi, s.gj))

        logger.info(f"Frame {self.frame} written to file.")
        self.frame += 1 # Increment frame counter
        
    def solve(self, ctx):
        # Nothing to do here
        pass

@nb.njit(parallel=True, cache=True)
def init_velocity_field(yvx, vx, xvy, vy, xsize_global, ysize_global):
    ny, nx = vx.shape
    for i in nb.prange(ny):
        for j in range(nx):
            vx[i, j] = (-(yvx[i]/ysize_global - 0.5)) * 1e-7
            vy[i, j] = (xvy[j]/xsize_global - 0.5) * 1e-7