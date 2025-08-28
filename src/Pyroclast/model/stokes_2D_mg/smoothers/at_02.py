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
Attempt 1: Reimplement the fully staggered loop with correct staggering.
"""

@nb.njit(cache=True, parallel=True)
def velocity_smoother_rb_gs(nx1: int, ny1: int,
                            dx: float, dy: float,
                            etap: np.ndarray, etab: np.ndarray,
                            vx: np.ndarray, vy: np.ndarray,
                            relax_v: float, BC: float,
                            vx_rhs: np.ndarray, vy_rhs: np.ndarray, max_iter: int,
                            th: int, cache_a: int, step_size: int):
    # Fast Implementation for big problems
    if th * cache_a > ny1 - 2:
        for _ in range(max_iter):
            # Work Split
            lim = (ny1 - 2) / cache_a

            for b in nb.prange(lim):
                start_y = cache_a * b + 1
                end_y = (ny1 - 1) if b + 1 == lim else cache_a * (b + 1) + 1

                # Iterate through j
                for j in range(1, nx1 - 1 + step_size * 3):
                    for i in range(start_y, end_y):

                        # Red Pass vx
                        jloc0 = j
                        if (i + jloc0) % 2 == 0 and 1 <= jloc0 <= nx1 - 2 and 1 <= i <= ny1 - 1:
                            vx[i, jloc0] = inline_loop_body_vx(i=i, j=jloc0, dx=dx, dy=dy, relax_v=relax_v,
                                                               etap=etap, etab=etab,
                                                               vx=vx, vy=vy, rhs=vx_rhs)

                        # Black Pass vx
                        jloc1 = j - step_size
                        if (i + jloc1) % 2 == 1 and 1 <= jloc1 <= nx1 - 2 and 1 <= i <= ny1 - 1:
                            vx[i, jloc1] = inline_loop_body_vx(i=i, j=jloc1, dx=dx, dy=dy, relax_v=relax_v,
                                                               etap=etap, etab=etab,
                                                               vx=vx, vy=vy, rhs=vx_rhs)

                        # Red Pass vy
                        jloc2 = j - (step_size * 2)
                        if (i + jloc2) % 2 == 0 and 1 <= jloc2 <= nx1 - 2 and 1 <= i <= ny1 - 1:
                                vy[i, jloc2] = inline_loop_body_vy(i=i, j=jloc2, dx=dx, dy=dy, relax_v=relax_v,
                                                                   etap=etap, etab=etab,
                                                                   vx=vx, vy=vy, rhs=vy_rhs)

                        # Black Pass vy
                        jloc3 = j - (step_size * 3)
                        if (i + jloc3) % 2 == 0 and 1 <= jloc3 <= nx1 - 2 and 1 <= i <= ny1 - 1:
                                vy[i, jloc3] = inline_loop_body_vy(i=i, j=jloc3, dx=dx, dy=dy, relax_v=relax_v,
                                                                   etap=etap, etab=etab,
                                                                   vx=vx, vy=vy, rhs=vy_rhs)


            apply_vx_BC(vx, BC)
            apply_vy_BC(vy, BC)

    #
    else:
        velocity_smoother_rb_gs(nx1=nx1, ny1=ny1,
                                dx=dx, dy=dy,
                                etap=etap, etab=etab,
                                vx=vx, vy=vy,
                                relax_v=relax_v, BC=BC,
                                vx_rhs=vx_rhs, vy_rhs=vy_rhs, max_iter=max_iter)