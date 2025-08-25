import numba as nb
import numpy as np

from Pyroclast.model.stokes_2D_mg.utils import apply_vy_BC
from ._inline_vy import compute_coeffs, compute_neighbor_sum


@nb.njit(cache=True, parallel=True)
def _vy_red_black_gs_sweep(nx1: int, ny1: int,
                           dx: float, dy: float,
                           etap: np.ndarray, etab: np.ndarray,
                           vx: np.ndarray, vy: np.ndarray,
                           relax_v: float, rhs: np.ndarray, BC: float) -> np.ndarray:
    """
    In-place Red-Black Gauss-Seidel update for vy.
    """
    # ----------------------------
    #  Red pass
    # ----------------------------
    for i in nb.prange(1, ny1 - 2):
        j_start = 1 if i % 2 == 0 else 2  # Red pass starts on even (i+j)
        for j in range(j_start, nx1 - 2, 2):
            vy_c1, vy_c2, vy_c3, vy_c4, vy_c5, vx_c1, vx_c2, vx_c3, vx_c4 = compute_coeffs(i, j, dx, dy, etap, etab)

            vy[i, j] = compute_neighbor_sum(i=i, j=j, relax_v=relax_v,
                                            vx=vx, vy=vy, rhs=rhs,
                                            vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4, vy_c5=vy_c5,
                                            vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4)

    # ----------------------------
    #  Black pass
    # ----------------------------
    for i in nb.prange(1, ny1 - 1):
        j_start = 2 if i % 2 == 0 else 1  # Black pass starts on odd (i+j)
        for j in range(j_start, nx1 - 2, 2):
            vy_c1, vy_c2, vy_c3, vy_c4, vy_c5, vx_c1, vx_c2, vx_c3, vx_c4 = compute_coeffs(i, j, dx, dy, etap, etab)

            vy[i, j] = compute_neighbor_sum(i=i, j=j, relax_v=relax_v,
                                            vx=vx, vy=vy, rhs=rhs,
                                            vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4, vy_c5=vy_c5,
                                            vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4)

    # Apply vy boundary conditions
    apply_vy_BC(vy, BC)

    return vy


# TODO add benchmark