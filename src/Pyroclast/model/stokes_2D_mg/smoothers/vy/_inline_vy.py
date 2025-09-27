"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/model/stokes_2D_mg/smoothers/vy/inline_vy.py 
Description: File contains loop body for vy pass to be inlined in sweeps.

Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


import numba
import numba as nb
import numpy as np
import importlib.util


def base_compute_coeffs_vy(i: int, j: int,
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


def base_compute_neighbor_sum_vy(i: int, j: int, relax_v: float,
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


cpu_compute_coeffs_vy = numba.njit(cache=True, inline="always")(base_compute_coeffs_vy)
cpu_compute_neighbor_sum_vy = numba.njit(cache=True, inline="always")(base_compute_neighbor_sum_vy)

# Setting defaults for gpu
gpu_inline_loop_body_vy = gpu_compute_neighbor_sum_vy = gpu_compute_coeffs_vy = None


@nb.njit(cache=True, inline="always")
def cpu_inline_loop_body_vy(i: int, j: int,
                            dx: float, dy: float, relax_v: float,
                            etap: np.ndarray, etab: np.ndarray,
                            vx: np.ndarray, vy: np.ndarray, rhs: np.ndarray) -> float:
    """
    Contains full loop body for vx pass.
    """
    vy_c1, vy_c2, vy_c3, vy_c4, vy_c5, vx_c1, vx_c2, vx_c3, vx_c4 = cpu_compute_coeffs_vy(i=i, j=j,
                                                                                          dx=dx, dy=dy,
                                                                                          etap=etap, etab=etab)

    return cpu_compute_neighbor_sum_vy(i=i, j=j, relax_v=relax_v,
                                       vx=vx, vy=vy, rhs=rhs,
                                       vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4,
                                       vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4, vy_c5=vy_c5)


@nb.njit(cache=True, parallel=True, inline="always")
def cpu_prep_vy_cache(nx1: int, ny1: int,
                      dx: float, dy: float,
                      etap: np.ndarray, etab: np.ndarray,
                      vy_cache: np.ndarray) -> np.ndarray:
    for i in nb.prange(1, ny1 - 1):
        for j in range(1, nx1 - 2):
            coeff_vec = np.array(cpu_compute_coeffs_vy(i, j, dx, dy, etap, etab))
            vy_cache[i, j] = coeff_vec


    return vy_cache


NUMBA_CUDA_AVAIL = importlib.util.find_spec("numba_cuda") is not None


# If Numba Cuda is available, prepare the decorators for the gpu exports as well.
if NUMBA_CUDA_AVAIL:
    import numba_cuda.numba.cuda as cuda

    # Defining base functions by decorating them with cuda
    gpu_compute_coeffs_vy = cuda.jit(cache=True, inline="always", device=True)(base_compute_coeffs_vy)
    gpu_compute_neighbor_sum_vy = cuda.jit(cache=True, inline="always", device=True)(base_compute_neighbor_sum_vy)


    @cuda.jit(cache=True, inline="always", device=True)
    def gpu_inline_loop_body_vy(i: int, j: int,
                                dx: float, dy: float, relax_v: float,
                                etap: np.ndarray, etab: np.ndarray,
                                vx: np.ndarray, vy: np.ndarray, rhs: np.ndarray) -> float:
        """
        Contains full loop body for vx pass.
        """
        vy_c1, vy_c2, vy_c3, vy_c4, vy_c5, vx_c1, vx_c2, vx_c3, vx_c4 = cpu_compute_coeffs_vy(i=i, j=j,
                                                                                              dx=dx, dy=dy,
                                                                                              etap=etap, etab=etab)

        return cpu_compute_neighbor_sum_vy(i=i, j=j, relax_v=relax_v,
                                           vx=vx, vy=vy, rhs=rhs,
                                           vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4,
                                           vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4, vy_c5=vy_c5)
