import numpy as np
import numba as nb


from Pyroclast.model.stokes_2D_mg import IncompressibleStokes2DMG
from Pyroclast.logging import get_logger

logger = get_logger(__name__)


class ThermoMechanical2D(IncompressibleStokes2DMG):
    def __init__(self, ctx):
        super().__init__(ctx)

    def solve(self, ctx):
        # Step 1: Solve the Stokes problem
        super().solve(ctx)

        # Step 2: Solve the thermal problem
        s, p, o = ctx
        T_new = np.empty_like(s.T0) # New temperature array
        T = s.T0.copy()             # Use previous timestep temperature as initial guess

        T_BC_TOP = 1573.0     # Top boundary condition, K
        T_BC_BOTTOM = 1573.0  # Bottom boundary condition, K

        # Perform Jacobi sweeps for thermal equation
        max_iter = 500
        for it in range(max_iter):
            jacobi_sweep_thermal(
                T_new,
                T,
                s.T0,
                s.kvx,
                s.kvy,
                s.rhocp,
                s.dx,
                s.dy,
                s.dt,
                omega=0.8
            )
            # Swap references for next iteration
            T, T_new = T_new, T

            # Enforce boundary conditions
            # Top boundary: (T[0, :] + T[1, :])/2 = T_BC_TOP
            T[0, :] = 2 * T_BC_TOP - T[1, :]

            # Bottom boundary: (T[-1, :] + T[-2, :])/2 = T_BC_BOTTOM
            T[-1, :] = 2 * T_BC_BOTTOM - T[-2, :]

            # Left and right boundaries Neumann (insulating)
            T[:, 0] = T[:, 1]
            T[:, -1] = T[:, -2]

            # --- NEW: compute and print residual ---
            res = thermal_residual(
                T,
                s.T0,
                s.kvx,
                s.kvy,
                s.rhocp,
                s.dx,
                s.dy,
                s.dt
            )
            print(f"[Thermal Jacobi] iter {it:4d}  residual = {res:.6e}")
            # ---------------------------------------

        # Copy final result back to state
        s.T0[:, :] = T[:, :]


    def dump(self, ctx):
        s, p, o = ctx

        # Dump state to file
        with open(f"frame_{str(self.frame).zfill(self.zpad)}.npz", 'wb') as f:
            np.savez(f, vx=s.vx, vy=s.vy, p=s.p,
                    rho=s.rho, etab=s.etab, etap=s.etap,
                    T=s.T0, kx=s.kvx, ky=s.kvy, alpha=s.alpha, rhocp=s.rhocp
                    )
        
        logger.info(f"Frame {self.frame} written to file.")
        self.frame += 1 # Increment frame counter



@nb.njit(cache=True, fastmath=True, parallel=True)
def jacobi_sweep_thermal(
    T_new,    # (ny1, nx1) output: new iterate
    T,        # (ny1, nx1) current iterate
    T0,    # (ny1, nx1) previous timestep solution
    kvx,      # (ny1, nx1) conductivity at vertical faces
    kvy,      # (ny1, nx1) conductivity at horizontal faces
    rhocp,    # (ny1, nx1)
    dx, dy, dt,
    omega     # weighted Jacobi parameter
):
    ny1, nx1 = T.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    # Interior update
    for i in nb.prange(1, ny1 - 1):
        for j in range(1, nx1 - 1):

            # Face conductivities (matching Matlab indexing)
            kvx1 = kvx[i, j-1]
            kvx2 = kvx[i, j]
            kvy1 = kvy[i-1, j]
            kvy2 = kvy[i, j]

            # Diagonal term
            A_diag = (
                rhocp[i, j] / dt
                + (kvx1 + kvx2) * inv_dx2
                + (kvy1 + kvy2) * inv_dy2
            )

            # Right-hand side: only time derivative term
            rhs = (rhocp[i, j] / dt) * T0[i, j]

            # Off-diagonal contribution
            sum_nb = (
                (-kvx1 * inv_dx2) * T[i, j-1]
                + (-kvx2 * inv_dx2) * T[i, j+1]
                + (-kvy1 * inv_dy2) * T[i-1, j]
                + (-kvy2 * inv_dy2) * T[i+1, j]
            )

            # Jacobi core update
            core = (rhs - sum_nb) / A_diag

            # Weighted Jacobi update
            Tij = T[i, j]
            T_new[i, j] = Tij + omega * (core - Tij)



@nb.njit(cache=True, fastmath=True, parallel=True)
def thermal_residual(
    T,        # (ny1, nx1) current iterate
    T0,       # (ny1, nx1) previous timestep solution
    kvx,      # (ny1, nx1) conductivity at vertical faces
    kvy,      # (ny1, nx1) conductivity at horizontal faces
    rhocp,    # (ny1, nx1)
    dx, dy, dt
):
    ny1, nx1 = T.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    res2 = 0.0
    npts = 0

    for i in nb.prange(1, ny1 - 1):
        for j in range(1, nx1 - 1):

            kvx1 = kvx[i, j-1]
            kvx2 = kvx[i, j]
            kvy1 = kvy[i-1, j]
            kvy2 = kvy[i, j]

            A_diag = (
                rhocp[i, j] / dt
                + (kvx1 + kvx2) * inv_dx2
                + (kvy1 + kvy2) * inv_dy2
            )

            rhs = (rhocp[i, j] / dt) * T0[i, j]

            sum_nb = (
                (-kvx1 * inv_dx2) * T[i, j-1]
                + (-kvx2 * inv_dx2) * T[i, j+1]
                + (-kvy1 * inv_dy2) * T[i-1, j]
                + (-kvy2 * inv_dy2) * T[i+1, j]
            )

            # Residual r = rhs - (A*T)
            r = rhs - (sum_nb + A_diag * T[i, j])

            res2 += r * r
            npts += 1

    if npts > 0:
        return np.sqrt(res2 / npts)
    else:
        return 0.0
