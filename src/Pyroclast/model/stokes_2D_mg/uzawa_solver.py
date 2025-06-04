"""High level Uzawa iteration using the velocity multigrid solver."""

import numpy as np

from .multigrid import Multigrid
from .smoother import pressure_sweep
from .mg_routines import uzawa_velocity_rhs
from .implicit_operators import p_residual, vx_residual, vy_residual
from .anderson import AndersonAccelerator
from .utils import apply_BC
from .viscosity_rescaler import ViscosityRescaler


class UzawaSolver:
    """Solve the Stokes system using Uzawa iterations and multigrid."""

    def __init__(self, ctx, levels, scaling=2.0):
        self.ctx = ctx
        self.mg = Multigrid(ctx, levels, scaling)
        self.fine = self.mg.hierarchy[0]
        self.rescaler = ViscosityRescaler(ctx, self.mg.hierarchy)
        self.relax_p = ctx.params.get("relax_p", 0.7)
        self.p_ref = ctx.params.get("p_ref", None)
        self.BC = ctx.params.BC
        self.state_k = np.zeros((3, self.fine.ny1, self.fine.nx1))
        self.state_next = np.zeros_like(self.state_k)
        self.accel = AndersonAccelerator(m=30, shape=(self.fine.ny1, self.fine.nx1))

    def compute_residuals(self, p_rhs, vx_rhs, vy_rhs):
        p_res = p_residual(self.fine.nx1, self.fine.ny1,
                           self.fine.dx, self.fine.dy,
                           self.fine.vx, self.fine.vy,
                           self.p_res, p_rhs)

        vx_res = vx_residual(self.fine.nx1, self.fine.ny1,
                             self.fine.dx, self.fine.dy,
                             self.fine.etap, self.fine.etab,
                             self.fine.vx, self.fine.vy,
                             self.p, self.fine.vx_res, vx_rhs)

        vy_res = vy_residual(self.fine.nx1, self.fine.ny1,
                             self.fine.dx, self.fine.dy,
                             self.fine.etap, self.fine.etab,
                             self.fine.vx, self.fine.vy,
                             self.p, self.fine.vy_res, vy_rhs)

        return p_res, vx_res, vy_res

    def solve(self, p_rhs, vx_rhs, vy_rhs,
              p_guess=None, vx_guess=None, vy_guess=None,
              max_cycles=50, tol=1e-7,
              nu1=3, nu2=3, gamma=1):
        self.p = np.zeros((self.fine.ny1, self.fine.nx1), dtype=np.float64)
        self.p_res = np.zeros((self.fine.ny1, self.fine.nx1), dtype=np.float64)

        if p_guess is not None:
            self.p[...] = p_guess
        if vx_guess is not None:
            self.fine.vx[...] = vx_guess
        if vy_guess is not None:
            self.fine.vy[...] = vy_guess

        for cycle in range(max_cycles):
            print(f"Cycle: {cycle}")

            self.fine.vx_rhs, self.fine.vy_rhs = uzawa_velocity_rhs(
                self.fine.nx1, self.fine.ny1,
                self.fine.dx, self.fine.dy,
                vx_rhs, vy_rhs, self.p,
                self.fine.vx_rhs, self.fine.vy_rhs)

            self.state_k[0] = self.fine.vx
            self.state_k[1] = self.fine.vy
            self.state_k[2] = self.p

            for _ in range(2):
                vx, vy = self.mg.vcycle(0, nu1, nu2, gamma)

            self.p = pressure_sweep(self.fine.nx1, self.fine.ny1,
                                    self.fine.dx, self.fine.dy,
                                    vx, vy, self.p,
                                    self.fine.etap,
                                    self.relax_p, p_rhs,
                                    p_ref=self.p_ref)

            self.state_next[0] = vx
            self.state_next[1] = vy
            self.state_next[2] = self.p
            state_accel = self.accel.update(self.state_k, self.state_next)
            if state_accel is not None:
                self.fine.vx[:, :] = state_accel[0]
                self.fine.vy[:, :] = state_accel[1]
                self.p[:, :] = state_accel[2]

                dp = self.p_ref - self.p[1, 1]
                self.p += dp
                apply_BC(self.p, self.fine.vx, self.fine.vy, self.BC)

            p_res, vx_res, vy_res = self.compute_residuals(p_rhs, vx_rhs, vy_rhs)

            dEta = self.fine.viscosity_contrast()

            N = np.sqrt(self.fine.nx1 * self.fine.ny1)
            p_res_rmse = np.linalg.norm(p_res) / N
            vx_res_rmse = np.linalg.norm(vx_res) / (N * dEta)
            vy_res_rmse = np.linalg.norm(vy_res) / (N * dEta)

            print(
                f"RMSE residuals: p = {p_res_rmse:.2e}, "
                f"vx = {vx_res_rmse:.2e}, vy = {vy_res_rmse:.2e}")

            if max(p_res_rmse, vx_res_rmse, vy_res_rmse) < tol and \
               self.rescaler.done_rescaling():
                break

            if p_res_rmse < 1e-15 and vx_res_rmse < 1e-5 and \
               vy_res_rmse < 1e-5 and self.rescaler.done_rescaling():
                print("Converged...")
                break

            if self.rescaler.update_viscosity():
                self.accel.reset()

        return self.p, self.fine.vx, self.fine.vy

