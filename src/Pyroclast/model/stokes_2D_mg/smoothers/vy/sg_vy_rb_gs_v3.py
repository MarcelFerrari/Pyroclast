"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/model/stokes_2D_mg/smoothers/vy/sg_vy_rb_gs_v3.py 
Description: In this file we aim to test if the staggered approach with explicit setup of the pipeline and explicit 
             tear down of the pipeline is faster than compiling an if into the the for loop.

Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import os.path
from typing import Type

import numba as nb
import numpy as np

from Pyroclast.model.stokes_2D_mg.utils import apply_vx_BC, apply_vy_BC
from ._inline_vy import compute_coeffs, compute_neighbor_sum


@nb.njit(cache=True, parallel=True)
def _vy_rb_gs_sweep(nx1, ny1,
                    dx, dy,
                    etap, etab,
                    vx, vy,
                    relax_v, rhs, BC,
                    th: int) -> np.ndarray:
    """
    In-place Red-Black Gauss-Seidel update for vx.
    """
    # INFO: Numba does not support increment by != 1
    # Setup pipeline with the first two rows of the red pass or red-black gauss-seidel
    # Less cells than threads
    if th < (nx1 - 4) / 2:
        i = 1
        j_start = 1 if i % 2 == 0 else 2  # Red pass starts on even (i+j)
        for j in range(j_start, nx1 - 2, 2):
            vy_c1, vy_c2, vy_c3, vy_c4, vy_c5, vx_c1, vx_c2, vx_c3, vx_c4 = compute_coeffs(i, j, dx, dy, etap, etab)

            vy[i, j] = compute_neighbor_sum(i=i, j=j, relax_v=relax_v,
                                            vx=vx, vy=vy, rhs=rhs,
                                            vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4, vy_c5=vy_c5,
                                            vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4)

        i = 1 + 1
        j_start = 1 if i % 2 == 0 else 2  # Red pass starts on even (i+j)
        for j in range(j_start, nx1 - 2, 2):
            vy_c1, vy_c2, vy_c3, vy_c4, vy_c5, vx_c1, vx_c2, vx_c3, vx_c4 = compute_coeffs(i, j, dx, dy, etap, etab)

            vy[i, j] = compute_neighbor_sum(i=i, j=j, relax_v=relax_v,
                                            vx=vx, vy=vy, rhs=rhs,
                                            vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4, vy_c5=vy_c5,
                                            vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4)
    else:
        for p in nb.prange(th):
            start = p * (nx1 - 3) / th
            end = (nx1 - 2) if p + 1 == th else (p + 1) * (nx1 - 3) / th

            for i in range(start, end):
                for j in range(1, 3):

                    # Abort if we have odd pairing
                    if i + j != 0:
                        continue

                    vy_c1, vy_c2, vy_c3, vy_c4, vy_c5, vx_c1, vx_c2, vx_c3, vx_c4 = compute_coeffs(i, j, dx, dy, etap,
                                                                                                   etab)

                    vy[i, j] = compute_neighbor_sum(i=i, j=j, relax_v=relax_v,
                                                    vx=vx, vy=vy, rhs=rhs,
                                                    vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4, vy_c5=vy_c5,
                                                    vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4)

    for i in nb.prange(1 + 2, ny1 - 1):
        # assert 1 <= i < ny1 -1, "Error with bounds - red pass"
        j_start = 1 if i % 2 == 0 else 2  # Red pass starts on even (i+j)
        for j in range(j_start, nx1 - 2, 2):
            vy_c1, vy_c2, vy_c3, vy_c4, vy_c5, vx_c1, vx_c2, vx_c3, vx_c4 = compute_coeffs(i, j, dx, dy, etap, etab)

            vy[i, j] = compute_neighbor_sum(i=i, j=j, relax_v=relax_v,
                                            vx=vx, vy=vy, rhs=rhs,
                                            vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4, vy_c5=vy_c5,
                                            vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4)

        # assert 1 <= i - 2 < ny1 -1, "Error with bounds - black pass"
        j_start = 2 if (i - 2) % 2 == 0 else 1  # Black pass starts on odd (i+j)
        for j in range(j_start, nx1 - 2, 2):
            vy_c1, vy_c2, vy_c3, vy_c4, vy_c5, vx_c1, vx_c2, vx_c3, vx_c4 = compute_coeffs(i - 2, j, dx, dy, etap, etab)

            vy[i - 2, j] = compute_neighbor_sum(i=i - 2, j=j, relax_v=relax_v,
                                                vx=vx, vy=vy, rhs=rhs,
                                                vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4, vy_c5=vy_c5,
                                                vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4)

    if th < (nx1 - 4) / 2:
        # Tear down pipeline with last two rows of the black pass
        i = nx1 - 1 - 2
        j_start = 2 if (i - 2) % 2 == 0 else 1  # Black pass starts on odd (i+j)
        for j in range(j_start, nx1 - 2, 2):
            vy_c1, vy_c2, vy_c3, vy_c4, vy_c5, vx_c1, vx_c2, vx_c3, vx_c4 = compute_coeffs(i, j, dx, dy, etap, etab)

            vy[i, j] = compute_neighbor_sum(i=i, j=j, relax_v=relax_v,
                                            vx=vx, vy=vy, rhs=rhs,
                                            vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4, vy_c5=vy_c5,
                                            vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4)

        i = nx1 - 1 - 1
        j_start = 2 if (i - 2) % 2 == 0 else 1  # Black pass starts on odd (i+j)
        for j in range(j_start, nx1 - 2, 2):
            vy_c1, vy_c2, vy_c3, vy_c4, vy_c5, vx_c1, vx_c2, vx_c3, vx_c4 = compute_coeffs(i, j, dx, dy, etap, etab)

            vy[i, j] = compute_neighbor_sum(i=i, j=j, relax_v=relax_v,
                                            vx=vx, vy=vy, rhs=rhs,
                                            vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4, vy_c5=vy_c5,
                                            vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4)

    else:
        for p in nb.prange(th):
            start = p * (nx1 - 3) / th
            end = (nx1 - 2) if p + 1 == th else (p + 1) * (nx1 - 3) / th

            for i in range(start, end):
                for j in range(1, 3):

                    # Abort if we have odd pairing
                    if i + j == 0:
                        continue

                    vy_c1, vy_c2, vy_c3, vy_c4, vy_c5, vx_c1, vx_c2, vx_c3, vx_c4 = compute_coeffs(i, j, dx, dy, etap,
                                                                                                   etab)

                    vy[i, j] = compute_neighbor_sum(i=i, j=j, relax_v=relax_v,
                                                    vx=vx, vy=vy, rhs=rhs,
                                                    vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4, vy_c5=vy_c5,
                                                    vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4)

    apply_vy_BC(vy, BC)

    return vy

# TODO add benchmark