import numba as nb
import numpy as np
import  os
from typing import Type
import math

from Pyroclast.model.stokes_2D_mg.smoothers.vx import inline_loop_body_vx
from Pyroclast.model.stokes_2D_mg.smoothers.vy import inline_loop_body_vy
from Pyroclast.model.stokes_2D_mg.utils import apply_vx_BC, apply_vy_BC
from Pyroclast.model.stokes_2D_mg.smoothers.base_rb_gs import velocity_smoother_rb_gs

"""
Attempt 9: Implemented Blocking Jacobi. And Thread Blocking with boundary jitter
"""

@nb.njit(cache=True, parallel=True)
def velocity_smoother_jacobi(nx1: int, ny1: int,
                             dx: float, dy: float,
                             etap: np.ndarray, etab: np.ndarray,
                             vx: np.ndarray, vy: np.ndarray,
                             relax_v: float, BC: float,
                             vx_rhs: np.ndarray, vy_rhs: np.ndarray, max_iter: int,
                             vx_new, vy_new,
                             th: int, cache_a: int, max_jitter: int = None) -> tuple[np.ndarray, np.ndarray]:
    if max_jitter is None:
        max_jitter = cache_a // 4

    # Fast Implementation for big problems
    if th * cache_a > nx1 - 2:
        for _ in range(max_iter):
            jitter = np.random.randint(-max_jitter, max_jitter, th)

            # Constrain jitter of 0 to be 0
            jitter[0] = 0

            for p in nb.prange(th):
                start_x = p * (nx1 - 2) // th + 1 + jitter[p]
                end_x = (nx1 - 1) if p + 1 == th else (p + 1) * (nx1 - 2) // th + 1 + jitter[p + 1]

                blocks = math.ceil((end_x - start_x) / cache_a)

                # Iterate through the cache blocks
                for b in range(blocks):
                    start_bx = start_x + b * cache_a
                    end_bx = end_x if b + 1 == blocks else start_x + (b + 1) * cache_a

                    # Iterate through j, add + 3 for offset for second pass, and vy pass
                    for i in range(1, ny1 - 1):
                        for j in range(start_bx, end_bx):
                            # Pass vx
                            if 1 <= j <= nx1 - 2 and 1 <= i <= ny1 - 1:
                                vx_new[i, j] = inline_loop_body_vx(i=i, j=j, dx=dx, dy=dy, relax_v=relax_v,
                                                                   etap=etap, etab=etab,
                                                                   vx=vx, vy=vy, rhs=vx_rhs)

                            # Pass vy
                            if 1 <= j <= nx1 - 1 and 1 <= i <= ny1 - 2:
                                vy_new[i, j] = inline_loop_body_vy(i=i, j=j, dx=dx, dy=dy, relax_v=relax_v,
                                                                   etap=etap, etab=etab,
                                                                   vx=vx, vy=vy, rhs=vy_rhs)

            apply_vx_BC(vx_new, BC)
            apply_vy_BC(vy_new, BC)
            vx, vx_new = vx_new, vx
            vy, vy_new = vy_new, vy

        return vx, vy

    else:
        velocity_smoother_rb_gs(nx1=nx1, ny1=ny1,
                                dx=dx, dy=dy,
                                etap=etap, etab=etab,
                                vx=vx, vy=vy,
                                relax_v=relax_v, BC=BC,
                                vx_rhs=vx_rhs, vy_rhs=vy_rhs, max_iter=max_iter)

        return vx, vy