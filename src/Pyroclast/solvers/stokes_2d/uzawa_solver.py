
"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: uzawa_solver.py
Description: This file implements the Uzawa solver for the Stokes flow
              and continuity equations in 2D.
                    
Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np

# Use the solver-agnostic multigrid base with Stokes-specific hooks
from Pyroclast.logging import get_logger
from Pyroclast.linalg import get_xp

from Pyroclast.solvers.anderson import AndersonAccelerator

from .grid_hierarchy import GridHierarchy
from .velocity_multigrid_2D import VelocityMultigrid2D
from .viscosity_rescaler import ViscosityRescaler

logger = get_logger(__name__)

class _UzawaSolverParams:
    """Helper class to extract Uzawa solver parameters from context."""

    def __init__(self, ctx):
        # Extract parameter namespace from context
        s, p, o = ctx
        self.relax_p = p.get("relax_p", 0.7)
        self.relax_v = p.get("relax_v", 0.7)
        self.uzawa_velocity_cycles = p.get("uzawa_velocity_cycles", 2)
        self.max_iterations = p.get("max_uzawa_iterations", 250)
        self.nu1 = p.get("mg_nu1", 5)
        self.nu2 = p.get("mg_nu2", 5)
        self.BC = p.BC
        self.res_tol = p.get("uzawa_stokes_res_tol", 1e-4)
        self.enable_gpu = p.get("enable_gpu", False)

class UzawaSolver:
    """Solve the Stokes system using Uzawa iterations and multigrid."""

    def __init__(self, ctx, nlevels, scaling=2.0):
        """Initialize the solver and allocate working arrays."""
        s, p, o = ctx
        
        # Set up solver parameters
        params = _UzawaSolverParams(ctx)
        self.relax_p = params.relax_p
        self.relax_v = params.relax_v
        self.nu1 = params.nu1
        self.nu2 = params.nu2
        self.velocity_cycles = params.uzawa_velocity_cycles
        self.max_cycles = params.max_iterations
        self.BC = params.BC  # type depends on your BC convention
        self.res_tol = params.res_tol
        self.enable_gpu = params.enable_gpu
        self.device = 'gpu' if self.enable_gpu else 'cpu'
        
        # Get xnp module
        xnp = get_xp(self.device)
        
        # Store material properties for the original Stokes problem
        self.stokes_etab = xnp.asarray(s.etab)
        self.stokes_etap = xnp.asarray(s.etap)
        self.stokes_rho = xnp.asarray(s.rho)

        # Set up multigrid solver for the velocity field
        self.hierarchy = GridHierarchy(ctx, nlevels, scaling)
        self.hierarchy.set_properties(self.stokes_etab,
                                      self.stokes_etap,
                                      self.stokes_rho)
        
        self.mg = VelocityMultigrid2D(self.hierarchy, scaling)
        self.fine = self.hierarchy[0] # Fine grid solution

        # Bind device-specific methods (CPU/GPU)
        self.bind_methods(device=self.device)
        
        # Read Resolution from fine grid
        # The pressure sub-problem should match this
        self.nx1 = self.fine.nx1
        self.ny1 = self.fine.ny1
        self.dx = self.fine.dx
        self.dy = self.fine.dy

        # For convenience, we extract the fine
        # grid solution from the velocity mg solver
        self.vx = self.fine.vx
        self.vx_res = self.fine.vx_res
        self.vx_rhs = self.fine.vx_rhs
        self.vy = self.fine.vy
        self.vy_res = self.fine.vy_res
        self.vy_rhs = self.fine.vy_rhs

        # Read material values from fine grid
        # This is important as we might be doing viscosity rescaling!
        self.etab = self.fine.etab
        self.etap = self.fine.etap
        self.rho = self.fine.rho

        # Set up viscosity rescaler
        self.rescaler = ViscosityRescaler(ctx, self.hierarchy)

        # Allocate memory for pressure solution and residual
        self.p = xnp.zeros((self.ny1, self.nx1))
        self.p_rhs = xnp.zeros((self.ny1, self.nx1))
        self.p_res = xnp.zeros((self.ny1, self.nx1))

        # Set up Anderson Acceleration
        self.accel = AndersonAccelerator(ctx, shape=(3, self.ny1, self.nx1))
        if self.accel.enabled:
            logger.info(f"Using Anderson Acceleration with m={self.accel.m}, "
                        f"beta={self.accel.beta}, reg={self.accel.reg}, "
                        f"scale_reg={self.accel.scale_reg}")
            self.state_k    = xnp.zeros((3, self.ny1, self.nx1))      # [vx, vy, p] at k
            self.state_next = xnp.zeros_like(self.state_k)            # G(x_k)

    def bind_methods(self, device):
        if device not in ['cpu', 'gpu']:
            raise ValueError(f"Unknown device: {device}")
        
        """Bind methods to the appropriate device (CPU/GPU)."""
        if device == 'cpu':
            from .smoothers.cpu import pressure_sweep
            self.pressure_sweep = pressure_sweep

            from .implicit_operators.cpu import uzawa_velocity_rhs, \
                                p_residual, vx_residual, vy_residual, \
                                compute_p_energy_norm, compute_vx_energy_norm, compute_vy_energy_norm
            self.uzawa_velocity_rhs = uzawa_velocity_rhs
            self.p_residual = p_residual
            self.vx_residual = vx_residual
            self.vy_residual = vy_residual
            self.compute_p_energy_norm = compute_p_energy_norm
            self.compute_vx_energy_norm = compute_vx_energy_norm
            self.compute_vy_energy_norm = compute_vy_energy_norm
            
            from .smoothers.cpu.bc import apply_BC
            self.apply_BC = apply_BC
        else:
            from .smoothers.gpu import pressure_sweep
            self.pressure_sweep = pressure_sweep

            from .implicit_operators.gpu import uzawa_velocity_rhs, \
                                p_residual, vx_residual, vy_residual, \
                                compute_p_energy_norm, compute_vx_energy_norm, compute_vy_energy_norm
            self.uzawa_velocity_rhs = uzawa_velocity_rhs
            self.p_residual = p_residual
            self.vx_residual = vx_residual
            self.vy_residual = vy_residual
            self.compute_p_energy_norm = compute_p_energy_norm
            self.compute_vx_energy_norm = compute_vx_energy_norm
            self.compute_vy_energy_norm = compute_vy_energy_norm
            from .smoothers.gpu.bc import apply_BC
            self.apply_BC = apply_BC

    def reset(self):
        """Reset the solver state (solution and residuals)."""
        self.p.fill(0.0)
        self.p_res.fill(0.0)
        self.hierarchy.reset()
        self.rescaler.reset()
        self.accel.reset()

    def compute_residuals(self, vx, vy, p):
        """Return residual arrays for pressure and velocity."""
        
        # Compute energy norm of residuals
        # Pressure residual
        self.p_res = self.p_residual(self.nx1, self.ny1,
                                self.dx, self.dy,
                                vx, vy,
                                self.p_res, self.stokes_p_rhs)
    
        # Velocity residuals
        self.vx_res = self.vx_residual(self.nx1, self.ny1,
                                    self.dx, self.dy,
                                    self.stokes_etap, self.stokes_etab,
                                    vx, vy, p,
                                    self.vx_res, self.stokes_vx_rhs)

        self.vy_res = self.vy_residual(self.nx1, self.ny1,
                                    self.dx, self.dy,
                                    self.stokes_etap, self.stokes_etab,
                                    vx, vy, p,
                                    self.vy_res, self.stokes_vy_rhs)

        # Compute energy norm residuals
        p_energy = self.compute_p_energy_norm(self.nx1, self.ny1, self.dx, self.dy, self.etap, self.p_res)
        vx_energy = self.compute_vx_energy_norm(self.nx1, self.ny1, self.dx, self.dy,
                                           self.stokes_etap, self.stokes_etab, self.vx_res)
        vy_energy = self.compute_vy_energy_norm(self.nx1, self.ny1, self.dx, self.dy,
                                           self.stokes_etap, self.stokes_etab, self.vy_res)
        
        # Gravity drives the flow only in vy direction
        # TODO: fix this ugly hack
        self.fine._w[...] = self.stokes_vy_rhs
        vy_rhs_norm = self.compute_vy_energy_norm(self.nx1, self.ny1, self.dx, self.dy,
                                           self.stokes_etap, self.stokes_etab, self.fine._w)
        
        residual = p_energy**2 + vx_energy**2 + vy_energy**2
        rhs_norm = vy_rhs_norm**2
        return np.sqrt(residual / (rhs_norm))

    def solve(self, stokes_p_rhs, stokes_vx_rhs, stokes_vy_rhs,
              p_guess=None, vx_guess=None, vy_guess=None):
        """Run Uzawa iterations until convergence."""

        # Get xp module
        xnp = get_xp(self.device)

        # Set up initial guess if given
        if p_guess is not None:
            p_guess = xnp.asarray(p_guess)
            self.p[...] = p_guess

            # Enforce zero-mean pressure on initial guess
            pbar = xnp.mean(self.p[1:-1, 1:-1])
            self.p -= pbar 
        if vx_guess is not None:
            vx_guess = xnp.asarray(vx_guess)
            self.vx[...] = vx_guess
        if vy_guess is not None:
            vy_guess = xnp.asarray(vy_guess)
            self.vy[...] = vy_guess

        # Store rhs for original stokes problem
        self.stokes_p_rhs = xnp.asarray(stokes_p_rhs)
        self.stokes_vx_rhs = xnp.asarray(stokes_vx_rhs)
        self.stokes_vy_rhs = xnp.asarray(stokes_vy_rhs)

        # Set up viscosity rescaler
        self.rescaler.set(self.stokes_etab, self.stokes_etap)

        for cycle in range(self.max_cycles):
            # Store current state for Anderson Acceleration
            if self.accel.enabled:
                self.state_k[0, ...] = self.vx
                self.state_k[1, ...] = self.vy
                self.state_k[2, ...] = self.p

            # Perform Uzawa Sweep

            # Update velocity right hand sides using the current pressure
            # This routine computes the uzawa rhs from the original stokes problem
            # Pressure here is assumed to be constant
            self.vx_rhs, self.vy_rhs = self.uzawa_velocity_rhs(self.nx1, self.ny1,
                                                            self.dx, self.dy,
                                                            self.stokes_vx_rhs, self.stokes_vy_rhs, self.p,
                                                            self.vx_rhs, self.vy_rhs)

            # Multigrid solve for velocity
            # This updates vx and vy in-place
            for _ in range(self.velocity_cycles):
                self.mg.vcycle(0, self.nu1, self.nu2)

            # Pressure update
            self.p = self.pressure_sweep(self.nx1, self.ny1,
                                    self.dx, self.dy,
                                    self.vx, self.vy,
                                    self.p, 
                                    self.etap,
                                    self.relax_p, self.stokes_p_rhs)

            # Attempt Anderson Acceleration
            if self.accel.enabled:
                self.state_next[0, ...] = self.vx
                self.state_next[1, ...] = self.vy
                self.state_next[2, ...] = self.p

                # Apply Anderson Acceleration
                # Note: we flatten the arrays to 1D for the accelerator
                x_acc = self.accel.update(self.state_k.reshape(-1),
                                           self.state_next.reshape(-1))
                if x_acc is not None:
                    # Readjust shapes
                    x_acc = x_acc.reshape(3, self.ny1, self.nx1)
                    self.vx[...] = x_acc[0, ...]
                    self.vy[...] = x_acc[1, ...]
                    self.p[...]  = x_acc[2, ...]

                    pbar = xnp.mean(self.p[1:-1, 1:-1])
                    self.p -= pbar  # Remove mean pressure drift
                    self.apply_BC(self.p, self.vx, self.vy, self.BC)

            
            # Compute residuals of the new solution
            res = self.compute_residuals(self.vx, self.vy, self.p)

            logger.debug((
                f"Cycle: {cycle}, "
                f"Relative Residual: {res:.3e}"
            ))

            # Update viscosity
            if self.rescaler.update_viscosity():
                self.accel.reset()  # Reset Anderson history if viscosity changed

            # Check convergence
            if res < self.res_tol and \
               self.rescaler.done_rescaling():
                logger.info(f"Uzawa solver converged in {cycle} cycles "
                            f"with relative residual {res:.3e}.")
                break

        if hasattr(xnp, 'asnumpy'):
            return xnp.asnumpy(self.p), \
                   xnp.asnumpy(self.vx), \
                   xnp.asnumpy(self.vy)
        else:
            return self.p, self.vx, self.vy