import numpy as np
from Pyroclast.logging import get_logger
from .smoothers.cpu.bc import apply_BC

logger = get_logger(__name__)
class IterativeRefinement:
    """
    Minimal iterative refinement wrapper around an existing UzawaSolver.

    Steps per refinement round:
      - compute residual wrt ORIGINAL RHS
      - solve correction system with RHS = residuals, zero initials
      - update solution: (p, vx, vy) += (dp, dvx, dvy)
      - print residual of updated solution
    """

    def __init__(self, uzawa_solver, ctx):
        self.solver = uzawa_solver
        self.ctx = ctx

    def run(self, p, vx, vy,
            stokes_p_rhs, stokes_vx_rhs, stokes_vy_rhs,
            p0=None, vx0=None, vy0=None,
            max_refine=5, tol=None):
        """
        Arguments:
          stokes_*_rhs: original RHS fields
          p0, vx0, vy0: optional initial guesses
          max_refine: number of refinement rounds
          tol: optional early-stop on normalized residual (uses solver.compute_residuals)
          print_fn: where to print per-round residuals
        Returns:
          (p, vx, vy) final fields from the wrapped UzawaSolver
        """
        if tol is None:
            tol = self.solver.res_tol

        # --- 1) Solve the original system once (with provided initial guesses) ---
        p[...], vx[...], vy[...] = \
        self.solver.solve(stokes_p_rhs, stokes_vx_rhs, stokes_vy_rhs,
                          p_guess=p0, vx_guess=vx0, vy_guess=vy0)

        # Compute + print residual of this solution (wrt ORIGINAL RHS)
        res = self.solver.compute_residuals(self.solver.vx, self.solver.vy, self.solver.p)
        logger.info(f"[initial solve] residual = {res:.3e}")
        if res < tol:
            return self.solver.p, self.solver.vx, self.solver.vy

        # --- 2) Refinement rounds ---
        rp = np.zeros_like(stokes_p_rhs)
        rvx = np.zeros_like(stokes_vx_rhs)
        rvy = np.zeros_like(stokes_vy_rhs)
        
        for k in range(1, max_refine + 1):
            # Set residuals as new RHS for the correction system
            rp[...] = self.solver.p_res
            rvx[...] = self.solver.vx_res
            rvy[...] = self.solver.vy_res

            # Reset solver and solve the CORRECTION system
            # RHS = residuals and zero initials
            self.solver.reset()
            self.solver.solve(rp, rvx, rvy)

            # (c) Correction is now in (p, vx, vy) of solver. Update the outer solution:
            p += self.solver.p
            vx += self.solver.vx
            vy += self.solver.vy

            pbar = np.mean(p[1:-1, 1:-1])
            p -= pbar  # Remove mean pressure drift
            apply_BC(p, vx, vy, self.ctx.params.BC)

            # (d) Restore ORIGINAL RHS on the solver and recompute + print residual
            self.solver.p[...] = p
            self.solver.vx[...] = vx
            self.solver.vy[...] = vy
            self.solver.stokes_p_rhs = stokes_p_rhs
            self.solver.stokes_vx_rhs = stokes_vx_rhs
            self.solver.stokes_vy_rhs = stokes_vy_rhs
            res = self.solver.compute_residuals(self.solver.vx, self.solver.vy, self.solver.p)

            logger.info(f"[refine {k}] residual = {res:.3e}")

            if res < tol:
                break

        return p, vx, vy
