import numba as nb
import numpy as np


@nb.njit(cache=True, inline="always")
def compute_coeffs(i: int, j: int,
                   dx: float, dy: float,
                   etap: np.ndarray, etab: np.ndarray):
    """
    External compute coeffs function to reduce on code duplication
    """
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

    return vx1_coeff, vx2_coeff, vx3_coeff, vx4_coeff, vx5_coeff, vy1_coeff, vy2_coeff, vy3_coeff, vy4_coeff


@nb.njit(cache=True, inline="always")
def compute_neighbor_sum(i: int, j: int, relax_v: float,
                         vx_c1: float, vx_c2: float, vx_c3: float, vx_c4: float, vx_c5: float,
                         vy_c1: float, vy_c2: float, vy_c3: float, vy_c4: float,
                         vx: np.ndarray, vy: np.ndarray, rhs: np.ndarray) -> float:

    # 3) Sum neighbor contributions
    sum_neighbors = (
            vx_c1 * vx[i, j - 1] +
            vx_c2 * vx[i - 1, j] +
            vx_c4 * vx[i + 1, j] +
            vx_c5 * vx[i, j + 1]
            +
            vy_c1 * vy[i - 1, j] +
            vy_c2 * vy[i, j] +
            vy_c3 * vy[i - 1, j + 1] +
            vy_c4 * vy[i, j + 1]
    )

    diag = vx_c3

    return (1.0 - relax_v) * vx[i, j] + relax_v * (rhs[i, j] - sum_neighbors) / diag


@nb.njit(cache=True, parallel=True, inline="always")
def prep_vx_cache(nx1: int, ny1: int,
                  dx: float, dy: float,
                  etap: np.ndarray, etab: np.ndarray,
                  vx_cache: np.ndarray) -> np.ndarray:
    for i in nb.prange(1, ny1 - 1):
        for j in range(1, nx1 - 2):
            coeff_vec = np.array(compute_coeffs(i, j, dx, dy, etap, etab))
            vx_cache[i, j] = coeff_vec

    return vx_cache