"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/model/stokes_2D_mg/smoothers/at_10.py 
Description: Attempt 10: Implemented Blocking Jacobi. And Thread Blocking. Cascading VY, Boundary Jitter

Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


import numba as nb
import numpy as np
import  os
from typing import Type
import math

from Pyroclast.model.stokes_2D_mg.smoothers.vx import cpu_inline_loop_body_vx
from Pyroclast.model.stokes_2D_mg.smoothers.vy import cpu_inline_loop_body_vy
from Pyroclast.model.stokes_2D_mg.utils import cpu_apply_vx_BC, cpu_apply_vy_BC
from Pyroclast.model.stokes_2D_mg.smoothers.base_rb_gs import velocity_smoother_rb_gs


@nb.njit(cache=True, parallel=True)
def velocity_smoother_jacobi(nx1: int, ny1: int,
                             dx: float, dy: float,
                             etap: np.ndarray, etab: np.ndarray,
                             vx: np.ndarray, vy: np.ndarray,
                             relax_v: float, BC: float,
                             vx_rhs: np.ndarray, vy_rhs: np.ndarray, max_iter: int,
                             vx_new: np.ndarray, vy_new: np.ndarray,
                             th: int, cache_a: int, step_size: int, max_jitter: int = None) -> tuple[np.ndarray, np.ndarray]:
    if max_jitter is None:
        max_jitter = cache_a // 4

    # Fast Implementation for big problems
    if th * cache_a > nx1 - 2:
        # Copy the results vx_new to vx
        vx[:] = vx_new[:]
        vy[:] = vy_new[:]

        for _ in range(max_iter // 2 * 2):
            # Work Split
            jitter = np.random.randint(-max_jitter, max_jitter, th)

            # Constrain jitter of 0 to be 0
            jitter[0] = 0

            for p in nb.prange(th):
                start_x = p * (nx1 - 2) // th + 1 + jitter[p]
                end_x = (nx1 - 1) if p + 1 == th else (p + 1) * (nx1 - 2) // th + 1 + jitter[p+1]

                blocks = math.ceil((end_x - start_x) / cache_a)

                # Iterate through the cache blocks
                for b in range(blocks):
                    start_bx = start_x + b * cache_a
                    end_bx = end_x if b + 1 == blocks else start_x + (b + 1) * cache_a

                    # Iterate through j, add + 3 for offset for second pass, and vy pass
                    for i in range(1, ny1 - 1 + step_size):
                        for j in range(start_bx, end_bx):
                            # Pass vx
                            iloc0 = i
                            if 1 <= j <= nx1 - 2 and 1 <= iloc0 <= ny1 - 1:
                                vx_new[iloc0, j] = cpu_inline_loop_body_vx(i=iloc0, j=j, dx=dx, dy=dy, relax_v=relax_v,
                                                                           etap=etap, etab=etab,
                                                                           vx=vx, vy=vy, rhs=vx_rhs)

                            # Pass vy
                            iloc1 = i - step_size
                            if 1 <= j <= nx1 - 1 and 1 <= iloc1 <= ny1 - 2:
                                vy_new[iloc1, j] = cpu_inline_loop_body_vy(i=iloc1, j=j, dx=dx, dy=dy, relax_v=relax_v,
                                                                           etap=etap, etab=etab,
                                                                           vx=vx_new, vy=vy, rhs=vy_rhs)

            # Copy the results vx_new to vx
            vx[:] = vx_new[:]
            vy[:] = vy_new[:]

            cpu_apply_vx_BC(vx, BC)
            cpu_apply_vy_BC(vy, BC)

        return vx, vy

    else:
        velocity_smoother_rb_gs(nx1=nx1, ny1=ny1,
                                dx=dx, dy=dy,
                                etap=etap, etab=etab,
                                vx=vx, vy=vy,
                                relax_v=relax_v, BC=BC,
                                vx_rhs=vx_rhs, vy_rhs=vy_rhs, max_iter=max_iter)

        return vx, vy
