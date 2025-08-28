import math

import numba as nb
import numpy as np

from Pyroclast.model.stokes_2D_mg.smoothers.base_rb_gs import velocity_smoother_rb_gs
from Pyroclast.model.stokes_2D_mg.smoothers.vx import inline_loop_body_vx
from Pyroclast.model.stokes_2D_mg.smoothers.vy import inline_loop_body_vy
from Pyroclast.model.stokes_2D_mg.utils import apply_vx_BC, apply_vy_BC

"""
Attempt 4: Implemented Staggered Jacobi. And Thread Blocking
"""

@nb.njit(cache=True, parallel=True)
def velocity_smoother_rb_gs(nx1: int, ny1: int,
                            dx: float, dy: float,
                            etap: np.ndarray, etab: np.ndarray,
                            vx: np.ndarray, vy: np.ndarray,
                            relax_v: float, BC: float,
                            vx_rhs: np.ndarray, vy_rhs: np.ndarray, max_iter: int,
                            th: int, cache_a: int, step_size: int,
                            vx_new: np.ndarray, vy_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Fast Implementation for big problems
    if th * cache_a > ny1 - 2:
        for _ in range(max_iter):
            # Work Split
            for p in nb.prange(th):
                start_y = p * (ny1 - 1) / th + 1
                end_y = (ny1 - 1) if p + 1 == th else (p + 1) * (ny1 - 1) / th + 1

                blocks = math.ceil((end_y - start_y) / cache_a)

                # Iterate through the cache blocks
                for b in range(blocks):
                    start_b = start_y + b * cache_a
                    end_b = end_y if b + 1 == blocks else start_y + (b + 1) * cache_a

                    # Iterate through j, add + 3 for offset for second pass, and vy pass
                    for j in range(1, nx1 - 1 + step_size):
                        for i in range(start_b, end_b):

                            # Red Pass vx
                            jloc0 = j
                            if (i + jloc0) % 2 == 0 and 1 <= jloc0 <= nx1 - 2 and 1 <= i <= ny1 - 1:
                                vx_new[i, jloc0] = inline_loop_body_vx(i=i, j=jloc0, dx=dx, dy=dy, relax_v=relax_v,
                                                                       etap=etap, etab=etab,
                                                                       vx=vx, vy=vy, rhs=vx_rhs)

                            # Red Pass vy
                            jloc2 = j - (step_size * 1)
                            if (i + jloc2) % 2 == 0 and 1 <= jloc2 <= nx1 - 2 and 1 <= i <= ny1 - 1:
                                    vy_new[i, jloc2] = inline_loop_body_vy(i=i, j=jloc2, dx=dx, dy=dy, relax_v=relax_v,
                                                                           etap=etap, etab=etab,
                                                                           vx=vx, vy=vy, rhs=vy_rhs)



            apply_vx_BC(vx_new, BC)
            apply_vy_BC(vy_new, BC)
            # Switch arrays
            vx, vx_new = vx_new, vx
            vy, vy_new = vy_new, vy

        return vx, vy
    #
    else:
        velocity_smoother_rb_gs(nx1=nx1, ny1=ny1,
                                dx=dx, dy=dy,
                                etap=etap, etab=etab,
                                vx=vx, vy=vy,
                                relax_v=relax_v, BC=BC,
                                vx_rhs=vx_rhs, vy_rhs=vy_rhs, max_iter=max_iter)

        return vx, vy