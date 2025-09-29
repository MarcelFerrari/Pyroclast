"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/model/stokes_2D_mg/smoothers/at_03.py 
Description: Attempt 0: Reimplement the fully staggered loop with correct staggering.

Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


import random

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
            lim = (ny1 - 2) // cache_a if (ny1-2) % cache_a != 0 else  (ny1 - 2) // cache_a + 1
            jitter = np.random.randint(low=0, high=cache_a // 2)

            for b in nb.prange(lim):
                start_y = max(cache_a * b + 1 - jitter, 1)
                end_y = (ny1 - 1) if b + 1 == lim else cache_a * (b + 1) + 1 - jitter

                # Iterate through j
                for j in range(1, nx1 - 1 + step_size * 3):
                    for i in range(start_y, end_y):

                        # Red Pass vx
                        jloc0 = j
                        if (i + jloc0) % 2 == 0 and 1 <= jloc0 <= nx1 - 2 and 1 <= i <= ny1 - 1:
                            vx[i, jloc0] = cpu_inline_loop_body_vx(i=i, j=jloc0, dx=dx, dy=dy, relax_v=relax_v,
                                                                   etap=etap, etab=etab,
                                                                   vx=vx, vy=vy, rhs=vx_rhs)

                        # Black Pass vx
                        jloc1 = j - step_size
                        if (i + jloc1) % 2 == 1 and 1 <= jloc1 <= nx1 - 2 and 1 <= i <= ny1 - 1:
                            vx[i, jloc1] = cpu_inline_loop_body_vx(i=i, j=jloc1, dx=dx, dy=dy, relax_v=relax_v,
                                                                   etap=etap, etab=etab,
                                                                   vx=vx, vy=vy, rhs=vx_rhs)

                        # Red Pass vy
                        jloc2 = j - (step_size * 2)
                        if (i + jloc2) % 2 == 0 and 1 <= jloc2 <= nx1 - 1 and 1 <= i <= ny1 - 2:
                                vy[i, jloc2] = cpu_inline_loop_body_vy(i=i, j=jloc2, dx=dx, dy=dy, relax_v=relax_v,
                                                                       etap=etap, etab=etab,
                                                                       vx=vx, vy=vy, rhs=vy_rhs)

                        # Black Pass vy
                        jloc3 = j - (step_size * 3)
                        if (i + jloc3) % 2 == 0 and 1 <= jloc3 <= nx1 - 1 and 1 <= i <= ny1 - 2:
                                vy[i, jloc3] = cpu_inline_loop_body_vy(i=i, j=jloc3, dx=dx, dy=dy, relax_v=relax_v,
                                                                       etap=etap, etab=etab,
                                                                       vx=vx, vy=vy, rhs=vy_rhs)


            cpu_apply_vx_BC(vx, BC)
            cpu_apply_vy_BC(vy, BC)

    #
    else:
        velocity_smoother_rb_gs(nx1=nx1, ny1=ny1,
                                dx=dx, dy=dy,
                                etap=etap, etab=etab,
                                vx=vx, vy=vy,
                                relax_v=relax_v, BC=BC,
                                vx_rhs=vx_rhs, vy_rhs=vy_rhs, max_iter=max_iter)
