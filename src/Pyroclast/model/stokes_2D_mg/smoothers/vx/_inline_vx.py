"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/model/stokes_2D_mg/smoothers/vx/inline_vx.py
Description: File contains shared inline routine for vx update

Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


import numba as nb
import numpy as np
import importlib.metadata


def base_compute_coeffs_vx(i: int, j: int,
                           dx: float, dy: float,
                           etap: np.ndarray, etab: np.ndarray):
    """
    External compute coeffs function to reduce on code duplication
    """
    # 1) Gather local viscosities
    etaA = etap[i, j]
    etaB = etap[i, j + 1]
    eta1 = etab[i - 1, j]
    eta2 = etab[i, j]

    # 2) Construct coefficients for x-momentum
    vx1_coeff = 2.0 * etaA / (dx * dx)
    vx2_coeff = eta1 / (dy * dy)
    vx3_coeff = -(eta1 + eta2) / (dy * dy) \
                - 2.0 * (etaA + etaB) / (dx * dx)
    vx4_coeff = eta2 / (dy * dy)
    vx5_coeff = 2.0 * etaB / (dx * dx)

    # Cross terms with vy
    vy1_coeff = eta1 / (dx * dy)
    vy2_coeff = -eta2 / (dx * dy)
    vy3_coeff = -eta1 / (dx * dy)
    vy4_coeff = eta2 / (dx * dy)

    # 22 fp mul
    # 2 fp add
    return vx1_coeff, vx2_coeff, vx3_coeff, vx4_coeff, vx5_coeff, vy1_coeff, vy2_coeff, vy3_coeff, vy4_coeff


def base_compute_neighbor_sum_vx(i: int, j: int, relax_v: float,
                                 vx_c1: float, vx_c2: float, vx_c3: float, vx_c4: float, vx_c5: float,
                                 vy_c1: float, vy_c2: float, vy_c3: float, vy_c4: float,
                                 vx: np.ndarray, vy: np.ndarray, rhs: np.ndarray) -> float:

    # 3) Sum neighbor contributions
    sum_neighbors = (
            vx_c1 * vx[i, j - 1] +
            vx_c2 * vx[i - 1, j] +
            vx_c4 * vx[i + 1, j] +
            vx_c5 * vx[i, j + 1]
            +
            vy_c1 * vy[i - 1, j] +
            vy_c2 * vy[i, j] +
            vy_c3 * vy[i - 1, j + 1] +
            vy_c4 * vy[i, j + 1]
    )

    diag = vx_c3

    # 11 fp mul
    # 7 fp add
    # 4 int add
    return (1.0 - relax_v) * vx[i, j] + relax_v * (rhs[i, j] - sum_neighbors) / diag


cpu_compute_coeffs_vx = nb.njit(cache=True, inline="always")(base_compute_coeffs_vx)
cpu_compute_neighbor_sum_vx = nb.njit(cache=True, inline="always")(base_compute_neighbor_sum_vx)


# Setting defaults for gpu
gpu_inline_loop_body_vx = gpu_compute_neighbor_sum_vx = gpu_compute_coeffs_vx = None


@nb.njit(cache=True, inline="always")
def cpu_inline_loop_body_vx(i: int, j: int,
                            dx: float, dy: float, relax_v: float,
                            etap: np.ndarray, etab: np.ndarray,
                            vx: np.ndarray, vy: np.ndarray, rhs: np.ndarray) -> float:
    """
    Contains full loop body for vx pass.
    """
    vx_c1, vx_c2, vx_c3, vx_c4, vx_c5, vy_c1, vy_c2, vy_c3, vy_c4 = cpu_compute_coeffs_vx(i=i, j=j,
                                                                                          dx=dx, dy=dy,
                                                                                          etap=etap, etab=etab)

    return cpu_compute_neighbor_sum_vx(i=i, j=j, relax_v=relax_v,
                                       vx=vx, vy=vy, rhs=rhs,
                                       vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4,
                                       vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4, vx_c5=vx_c5)


@nb.njit(cache=True, parallel=True, inline="always")
def cpu_prep_vx_cache(nx1: int, ny1: int,
                      dx: float, dy: float,
                      etap: np.ndarray, etab: np.ndarray,
                      vx_cache: np.ndarray) -> np.ndarray:
    for i in nb.prange(1, ny1 - 1):
        for j in range(1, nx1 - 2):
            coeff_vec = np.array(cpu_compute_coeffs_vx(i, j, dx, dy, etap, etab))
            vx_cache[i, j] = coeff_vec


    return vx_cache


NUMBA_CUDA_AVAIL = False


try:
    importlib.metadata.version("numba-cuda")
    NUMBA_CUDA_AVAIL = True
except importlib.metadata.PackageNotFoundError:
    pass


# If Numba Cuda is available, prepare the decorators for the gpu exports as well.
if NUMBA_CUDA_AVAIL:
    import numba.cuda as cuda

    # Defining base functions by decorating them with cuda
    gpu_compute_coeffs_vx = cuda.jit(cache=True, inline="always", device=True)(base_compute_coeffs_vx)
    gpu_compute_neighbor_sum_vx = cuda.jit(cache=True, inline="always", device=True)(base_compute_neighbor_sum_vx)


    @cuda.jit(cache=True, inline="always", device=True)
    def gpu_inline_loop_body_vx(i: int, j: int,
                                dx: float, dy: float, relax_v: float,
                                etap: np.ndarray, etab: np.ndarray,
                                vx: np.ndarray, vy: np.ndarray, rhs: np.ndarray) -> float:
        """
        Contains full loop body for vx pass.
        """
        vy_c1, vy_c2, vy_c3, vy_c4, vy_c5, vx_c1, vx_c2, vx_c3, vx_c4 = gpu_compute_coeffs_vx(i=i, j=j,
                                                                                              dx=dx, dy=dy,
                                                                                              etap=etap, etab=etab)

        return gpu_compute_neighbor_sum_vx(i=i, j=j, relax_v=relax_v,
                                           vx=vx, vy=vy, rhs=rhs,
                                           vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4,
                                           vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4, vy_c5=vy_c5)