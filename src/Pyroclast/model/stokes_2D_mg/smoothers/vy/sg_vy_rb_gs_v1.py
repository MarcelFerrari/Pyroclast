import numba as nb
import numpy as np

from Pyroclast.model.stokes_2D_mg.utils import cpu_apply_vy_BC
from ._inline_vy import cpu_inline_loop_body_vy

"""
In this file we aim to test if the staggered approach to red-black gauss seidel is faster than the regular approach.
We consider this a success if it is as fast as the non-staggered approach
"""


@nb.njit(cache=True, parallel=True)
def _vy_rb_gs_sweep(nx1, ny1,
                    dx, dy,
                    etap, etab,
                    vx, vy,
                    relax_v, rhs, BC) -> np.ndarray:
    """
    In-place Red-Black Gauss-Seidel update for vx.
    """
    for i in nb.prange(1, ny1 - 1 + 2):
        # Staggered Red pass
        if 1 <= i < ny1 - 1:
            j_start = 1 if i % 2 == 0 else 2  # Red pass starts on even (i+j)
            for j in range(j_start, nx1 - 2, 2):
                vy[i, j] = cpu_inline_loop_body_vy(i=i, j=j, relax_v=relax_v,
                                                   vx=vx, vy=vy, rhs=rhs,
                                                   dx=dx, dy=dy,
                                                   etab=etab, etap=etap)

        # Staggered black pass
        if 1 <= i - 2 <= ny1 - 1:
            j_start = 2 if (i - 2) % 2 == 0 else 1  # Black pass starts on odd (i+j)
            for j in range(j_start, nx1 - 2, 2):
                vy[i, j] = cpu_inline_loop_body_vy(i=i, j=j, relax_v=relax_v,
                                                   vx=vx, vy=vy, rhs=rhs,
                                                   dx=dx, dy=dy,
                                                   etab=etab, etap=etap)

    cpu_apply_vy_BC(vx, BC)

    return vx

# TODO add benchmark