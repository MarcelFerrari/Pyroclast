import numba as nb
import numpy as np


@nb.njit(cache=True, inline="always")
def compute_coeffs(i: int, j: int,
                   dx: float, dy: float,
                   etap: np.ndarray, etab: np.ndarray):
    # 1) Gather local viscosities
    etaA = etap[i, j]
    etaB = etap[i + 1, j]
    eta1 = etab[i, j - 1]
    eta2 = etab[i, j]

    # 2) Construct coefficients for y-momentum
    vy1_coeff = eta1 / (dx * dx)
    vy2_coeff = 2.0 * etaA / (dy * dy)
    vy3_coeff = -2.0 * etaA / (dy * dy) \
                - 2.0 * etaB / (dy * dy) \
                - eta1 / (dx * dx) \
                - eta2 / (dx * dx)
    vy4_coeff = 2.0 * etaB / (dy * dy)
    vy5_coeff = eta2 / (dx * dx)

    # Cross terms with vx
    vx1_coeff = eta1 / (dx * dy)
    vx2_coeff = -eta1 / (dx * dy)
    vx3_coeff = -eta2 / (dx * dy)
    vx4_coeff = eta2 / (dx * dy)

    return vy1_coeff, vy2_coeff, vy3_coeff, vy4_coeff, vy5_coeff, vx1_coeff, vx2_coeff, vx3_coeff, vx4_coeff


@nb.njit(cache=True, inline="always")
def compute_neighbor_sum(i: int, j: int, relax_v: float,
                         vy_c1: float, vy_c2: float, vy_c3: float, vy_c4: float, vy_c5: float,
                         vx_c1: float, vx_c2: float, vx_c3: float, vx_c4: float,
                         vx: np.ndarray, vy: np.ndarray, rhs: np.ndarray) -> float:
    # 3) Sum neighbor contributions
    sum_neighbors = (
        # vy neighbors
            vy_c1 * vy[i, j - 1] +
            vy_c2 * vy[i - 1, j] +
            vy_c4 * vy[i + 1, j] +
            vy_c5 * vy[i, j + 1]
            +
            # cross terms with vx
            vx_c1 * vx[i, j - 1] +
            vx_c2 * vx[i + 1, j - 1] +
            vx_c3 * vx[i, j] +
            vx_c4 * vx[i + 1, j]
    )

    diag = vy_c3

    # 4) Gauss-Seidel in-place update
    return (1.0 - relax_v) * vy[i, j] + relax_v * (rhs[i, j] - sum_neighbors) / diag


@nb.njit(cache=True, inline="always")
def inline_loop_body_vy(i: int, j: int,
                        dx: float, dy: float, relax_v: float,
                        etap: np.ndarray, etab: np.ndarray,
                        vx: np.ndarray, vy: np.ndarray, rhs: np.ndarray) -> float:
    """
    Contains full loop body for vx pass.
    """
    vy_c1, vy_c2, vy_c3, vy_c4, vy_c5, vx_c1, vx_c2, vx_c3, vx_c4 = compute_coeffs(i=i, j=j,
                                                                                   dx=dx, dy=dy,
                                                                                   etap=etap, etab=etab)

    return compute_neighbor_sum(i=i, j=j, relax_v=relax_v,
                                vx=vx, vy=vy, rhs=rhs,
                                vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4,
                                vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4, vy_c5=vy_c5)


@nb.njit(cache=True, parallel=True, inline="always")
def prep_vy_cache(nx1: int, ny1: int,
                  dx: float, dy: float,
                  etap: np.ndarray, etab: np.ndarray,
                  vy_cache: np.ndarray) -> np.ndarray:
    for i in nb.prange(1, ny1 - 1):
        for j in range(1, nx1 - 2):
            coeff_vec = np.array(compute_coeffs(i, j, dx, dy, etap, etab))
            vy_cache[i, j] = coeff_vec


    return vy_cache