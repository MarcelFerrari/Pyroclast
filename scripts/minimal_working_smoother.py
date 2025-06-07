import numba as nb
import numpy as np
from time import time

from numba.cpython.heapq import reversed_range

from benchmark.utils import dtf

# Set Numba threading layer to 'omp' for parallel execution
nb.config.THREADING_LAYER = 'omp'


# Jacobi update for vx
@nb.njit(cache=True, parallel=True)
def vx_jacobi_sweep(nx1: int, ny1: int,
                    dx: float, dy: float,
                    etap: np.ndarray, etab: np.ndarray,
                    vx: np.ndarray, vy: np.ndarray,
                    vx_new: np.ndarray,
                    relax_v:float, BC: float, rhs: np.ndarray):
    # Loop only over the interior cells
    for i in nb.prange(1, ny1 - 1):
        for j in range(1, nx1 - 2):
            # Gather local (i,j) viscosity coefficients
            etaA = etap[i, j]
            etaB = etap[i, j + 1]
            eta1 = etab[i - 1, j]
            eta2 = etab[i, j]

            # Construct coefficients for x-momentum
            vx1_coeff = 2.0 * etaA / (dx * dx)
            vx2_coeff = eta1 / (dy * dy)
            vx3_coeff = - (eta1 + eta2) / (dy * dy) - 2.0 * (etaA + etaB) / (dx * dx)
            vx4_coeff = eta2 / (dy * dy)
            vx5_coeff = 2.0 * etaB / (dx * dx)

            # Cross terms with vy
            vy1_coeff = eta1 / (dx * dy)
            vy2_coeff = -eta2 / (dx * dy)
            vy3_coeff = -eta1 / (dx * dy)
            vy4_coeff = eta2 / (dx * dy)

            # Sum neighbor contributions for Jacobi
            sum_neighbors = (
                    vx1_coeff * vx[i, j - 1] +
                    vx2_coeff * vx[i - 1, j] +
                    vx4_coeff * vx[i + 1, j] +
                    vx5_coeff * vx[i, j + 1]
                    +
                    vy1_coeff * vy[i - 1, j] +
                    vy2_coeff * vy[i, j] +
                    vy3_coeff * vy[i - 1, j + 1] +
                    vy4_coeff * vy[i, j + 1]
            )

            diag = vx3_coeff

            # Jacobi update
            vx_new[i, j] = (1.0 - relax_v) * vx[i, j] + \
                           relax_v * (rhs[i, j] - sum_neighbors) / diag

    # Apply boundary conditions
    vx_new[0, :] = - BC * vx_new[1, :]  # Top boundary
    vx_new[-1, :] = - BC * vx_new[-2, :]  # Bottom boundary
    vx_new[:, 0] = 0.0  # Left boundary
    vx_new[:, -2:] = 0.0  # Right boundary + ghost cell

    # Copy solution to vx
    vx[:, :] = vx_new[:, :]


# Gauss-Seidel update for vx
@nb.njit(cache=True)
def vx_gs_sweep(nx1, ny1,
                dx, dy,
                etap, etab,
                vx, vy,
                relax_v, rhs, BC):
    """
    In-place Red-Black Gauss-Seidel update for vx.
    """
    for i in range(1, ny1 - 1):
        for j in range(1, nx1 - 2):
            # 1) Gather local viscosities
            etaA = etap[i, j]
            etaB = etap[i, j + 1]
            eta1 = etab[i - 1, j]
            eta2 = etab[i, j]

            # 2) Construct coefficients for x-momentum
            vx1_coeff = 2.0 * etaA / (dx * dx)
            vx2_coeff = eta1 / (dy * dy)
            vx3_coeff = -(eta1 + eta2) / (dy * dy) \
                        - 2.0 * (etaA + etaB) / (dx * dx)
            vx4_coeff = eta2 / (dy * dy)
            vx5_coeff = 2.0 * etaB / (dx * dx)

            # Cross terms with vy
            vy1_coeff = eta1 / (dx * dy)
            vy2_coeff = -eta2 / (dx * dy)
            vy3_coeff = -eta1 / (dx * dy)
            vy4_coeff = eta2 / (dx * dy)

            # 3) Sum neighbor contributions
            sum_neighbors = (
                    vx1_coeff * vx[i, j - 1] +
                    vx2_coeff * vx[i - 1, j] +
                    vx4_coeff * vx[i + 1, j] +
                    vx5_coeff * vx[i, j + 1]
                    +
                    vy1_coeff * vy[i - 1, j] +
                    vy2_coeff * vy[i, j] +
                    vy3_coeff * vy[i - 1, j + 1] +
                    vy4_coeff * vy[i, j + 1]
            )

            diag = vx3_coeff

            # Gauss-Seidel in-place update
            vx[i, j] = (1.0 - relax_v) * vx[i, j] \
                       + relax_v * (rhs[i, j] - sum_neighbors) / diag

    return vx


# Red black Gauss-Seidel update for vx
@nb.njit(cache=True, parallel=True)
def vx_rb_gs_sweep(nx1, ny1,
                   dx, dy,
                   etap, etab,
                   vx, vy,
                   relax_v, rhs, BC):
    """
    In-place Red-Black Gauss-Seidel update for vx.
    """

    # ----------------------------
    #  Red pass: (i + j) % 2 == 0
    # ----------------------------
    for i in nb.prange(1, ny1 - 1):
        j_start = 1 if i % 2 == 0 else 2  # Red pass starts on even (i+j)
        for j in range(j_start, nx1 - 2, 2):
            # 1) Gather local viscosities
            etaA = etap[i, j]
            etaB = etap[i, j + 1]
            eta1 = etab[i - 1, j]
            eta2 = etab[i, j]

            # 2) Construct coefficients for x-momentum
            vx1_coeff = 2.0 * etaA / (dx * dx)
            vx2_coeff = eta1 / (dy * dy)
            vx3_coeff = -(eta1 + eta2) / (dy * dy) \
                        - 2.0 * (etaA + etaB) / (dx * dx)
            vx4_coeff = eta2 / (dy * dy)
            vx5_coeff = 2.0 * etaB / (dx * dx)

            # Cross terms with vy
            vy1_coeff = eta1 / (dx * dy)
            vy2_coeff = -eta2 / (dx * dy)
            vy3_coeff = -eta1 / (dx * dy)
            vy4_coeff = eta2 / (dx * dy)

            # 3) Sum neighbor contributions
            sum_neighbors = (
                    vx1_coeff * vx[i, j - 1] +
                    vx2_coeff * vx[i - 1, j] +
                    vx4_coeff * vx[i + 1, j] +
                    vx5_coeff * vx[i, j + 1]
                    +
                    vy1_coeff * vy[i - 1, j] +
                    vy2_coeff * vy[i, j] +
                    vy3_coeff * vy[i - 1, j + 1] +
                    vy4_coeff * vy[i, j + 1]
            )

            diag = vx3_coeff

            # Gauss-Seidel in-place update
            vx[i, j] = (1.0 - relax_v) * vx[i, j] \
                       + relax_v * (rhs[i, j] - sum_neighbors) / diag

    # ----------------------------
    #  Black pass: (i + j) % 2 == 1
    # ----------------------------
    for i in nb.prange(1, ny1 - 1):
        j_start = 2 if i % 2 == 0 else 1  # Black pass starts on odd (i+j)
        for j in range(j_start, nx1 - 2, 2):
            # 1) Gather local viscosities
            etaA = etap[i, j]
            etaB = etap[i, j + 1]
            eta1 = etab[i - 1, j]
            eta2 = etab[i, j]

            # 2) Construct coefficients for x-momentum
            vx1_coeff = 2.0 * etaA / (dx * dx)
            vx2_coeff = eta1 / (dy * dy)
            vx3_coeff = -(eta1 + eta2) / (dy * dy) \
                        - 2.0 * (etaA + etaB) / (dx * dx)
            vx4_coeff = eta2 / (dy * dy)
            vx5_coeff = 2.0 * etaB / (dx * dx)

            # Cross terms with vy
            vy1_coeff = eta1 / (dx * dy)
            vy2_coeff = -eta2 / (dx * dy)
            vy3_coeff = -eta1 / (dx * dy)
            vy4_coeff = eta2 / (dx * dy)

            # 3) Sum neighbor contributions

            sum_neighbors = (
                    vx1_coeff * vx[i, j - 1] +
                    vx2_coeff * vx[i - 1, j] +
                    vx4_coeff * vx[i + 1, j] +
                    vx5_coeff * vx[i, j + 1]
                    +
                    vy1_coeff * vy[i - 1, j] +
                    vy2_coeff * vy[i, j] +
                    vy3_coeff * vy[i - 1, j + 1] +
                    vy4_coeff * vy[i, j + 1]
            )

            diag = vx3_coeff

            # Gauss-Seidel in-place update
            vx[i, j] = (1.0 - relax_v) * vx[i, j] \
                       + relax_v * (rhs[i, j] - sum_neighbors) / diag

    return vx


if __name__ == "__main__":
    # Generate synthetic data for testing
    max_iter = 128
    nx = 2048
    ny = 2048
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    nx1 = nx + 1
    ny1 = ny + 1

    # Generate random material properties
    etab = np.random.rand(ny1, nx1) * 1e19 + 1e19  # Viscosity at basic nodes
    etap = np.random.rand(ny1, nx1) * 1e19 + 1e19  # Viscosity at pressure nodes

    # Generate solution and rhs arrays
    vx = np.zeros((ny1, nx1))
    vx_new = np.zeros((ny1, nx1))  # New array for Jacobi update
    vy = np.zeros((ny1, nx1))
    vx_rhs = np.zeros((ny1, nx1))

    BC = -1.0
    relax_v = 0.7

    n_samples = 15

    # Dry run for all smoothers
    vx_jacobi_sweep(nx1, ny1,
                    dx, dy,
                    etap, etab,
                    vx, vy,
                    vx_new,
                    relax_v=relax_v, BC=BC, rhs=vx_rhs)



    vx_gs_sweep(nx1, ny1,
                dx, dy,
                etap, etab,
                vx, vy,
                relax_v=relax_v, rhs=vx_rhs, BC=BC)

    vx_rb_gs_sweep(nx1, ny1,
                   dx, dy,
                   etap, etab,
                   vx, vy,
                   relax_v=relax_v, rhs=vx_rhs, BC=BC)

    # Measure performance
    ms_jacobi = []
    for s in range(n_samples):
        start = dtf()
        for i in range(max_iter):
            vx_jacobi_sweep(nx1, ny1,
                            dx, dy,
                            etap, etab,
                            vx, vy,
                            vx_new,
                            relax_v=relax_v, BC=BC, rhs=vx_rhs)
        end = dtf()
        ms_jacobi.append((end - start).total_seconds())
        print(f"Jacobi time, sample {s}: {(end - start).total_seconds():.4f} seconds for {max_iter} iterations")

    print(f"Average time: {np.mean(ms_jacobi):.4f} seconds, std: {np.std(ms_jacobi):.4f} seconds")

    ms_gs = []
    for s in reversed_range(n_samples):
        start = dtf()
        for i in range(max_iter):
            vx_gs_sweep(nx1, ny1,
                        dx, dy,
                        etap, etab,
                        vx, vy,
                        relax_v=relax_v, rhs=vx_rhs, BC=BC)
        end = dtf()
        ms_gs.append((end - start).total_seconds())
        print(f"Gauss-Seidel time, sample {s}: {(end - start).total_seconds():.4f} seconds for {max_iter} iterations")

    print(f"Average time: {np.mean(ms_gs):.4f} seconds, std: {np.std(ms_gs):.4f} seconds")

    ms_rb_gs = []
    for s in range(n_samples):
        start = dtf()
        for i in range(max_iter):
            vx_rb_gs_sweep(nx1, ny1,
                           dx, dy,
                           etap, etab,
                           vx, vy,
                           relax_v=relax_v, rhs=vx_rhs, BC=BC)
        end = dtf()
        ms_rb_gs.append((end - start).total_seconds())
        print(f"Red-Black Gauss-Seidel time, sample {s}: {(end - start).total_seconds():.4f} seconds for {max_iter} iterations")

    print(f"Average time: {np.mean(ms_rb_gs):.4f} seconds, std: {np.std(ms_rb_gs):.4f} seconds")