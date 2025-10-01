"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/model/stokes_2D_mg/smoothers/inline_routines.py
Description: File contains inline routines for velocity smoothers

Author: Marcel Ferrari, Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


import importlib.metadata
import os

import numba as nb
import numpy as np

use_fast_math_cpu = os.environ.get("PYROCLAST_FASTMATH_CPU", default=False)
use_fast_math_gpu = os.environ.get("PYROCLAST_FASTMATH_GPU", default=False)


def base_compute_coeffs_vx(i: int, j: int,
                           dx: float, dy: float,
                           etap: np.ndarray, etab: np.ndarray) \
        -> tuple[float, float, float, float, float, float, float, float, float]:
    """
    Fetch the coefficients based on the viscosity for the iteration of a given vx cell.

    :param i: Index in y direction.
    :param j: Index in x direction.
    :param dx: Step size in x direction.
    :param dy: Step size in y direction.
    :param etap: Viscosity on pressure nodes.
    :param etab: Viscosity on basic nodes.

    :returns coefficients for vx [1-5] according to literature, coefficients for vy [1-4] according to literature
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
    """
    Based on the coefficients, update the current cell of the vx grid.
    :param i: Index in y direction.
    :param j: Index in x direction.

    :param relax_v: Weight factor of new value and old value at given grid location.
    :param vx_c1: coeff vx 1
    :param vx_c2: coeff vx 2
    :param vx_c3: coeff vx 3
    :param vx_c4: coeff vx 4
    :param vx_c5: coeff vx 5

    :param vy_c1: coeff vy 1
    :param vy_c2: coeff vy 2
    :param vy_c3: coeff vy 3
    :param vy_c4: coeff vy 4

    :param vx: The current vx array
    :param vy: The current vy array

    :param rhs: The target right hand side of the equation

    :returns the updated value for vx (update of the array is subject to the choice of algorithm)
    """

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


def base_compute_coeffs_vy(i: int, j: int,
                           dx: float, dy: float,
                           etap: np.ndarray, etab: np.ndarray) \
        -> tuple[float, float, float, float, float, float, float, float, float]:
    """
    Fetch the coefficients based on the viscosity for the iteration of a given vy cell.

    :param i: Index in y direction.
    :param j: Index in x direction.
    :param dx: Step size in x direction.
    :param dy: Step size in y direction.
    :param etap: Viscosity on pressure nodes.
    :param etab: Viscosity on basic nodes.

    :returns coefficients for vy [1-5] according to literature, coefficients for vx [1-4] according to literature
    """
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
    """
    Based on the coefficients, update the current cell of the vx grid.
    :param i: Index in y direction.
    :param j: Index in x direction.

    :param relax_v: Weight factor of new value and old value at given grid location.
    :param vy_c1: coeff vy 1
    :param vy_c2: coeff vy 2
    :param vy_c3: coeff vy 3
    :param vy_c4: coeff vy 4
    :param vy_c5: coeff vy 5

    :param vx_c1: coeff vx 1
    :param vx_c2: coeff vx 2
    :param vx_c3: coeff vx 3
    :param vx_c4: coeff vx 4

    :param vx: The current vx array
    :param vy: The current vy array

    :param rhs: The target right hand side of the equation

    :returns the updated value for vy (update of the array is subject to the choice of algorithm)
    """
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


# Decorating functions and preparing them for export
cpu_compute_coeffs_vx = nb.njit(cache=True,
                                inline="always",
                                fastmath=use_fast_math_cpu)(base_compute_coeffs_vx)
cpu_compute_coeffs_vy = nb.njit(cache=True,
                                inline="always",
                                fastmath=use_fast_math_cpu)(base_compute_coeffs_vy)
cpu_compute_neighbor_sum_vx = nb.njit(cache=True,
                                      inline="always",
                                      fastmath=use_fast_math_cpu)(base_compute_neighbor_sum_vx)
cpu_compute_neighbor_sum_vy = nb.njit(cache=True,
                                      inline="always",
                                      fastmath=use_fast_math_cpu)(base_compute_neighbor_sum_vy)


@nb.njit(cache=True, inline="always", fastmath=use_fast_math_cpu)
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
                                       vy_c3=vy_c3, vy_c2=vy_c2, vy_c4=vy_c4, vy_c1=vy_c1,
                                       vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4, vx_c5=vx_c5)


@nb.njit(cache=True, inline="always", fastmath=use_fast_math_cpu)
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
    gpu_compute_coeffs_vx = cuda.jit(cache=True,
                                     inline="always",
                                     device=True,
                                     fastmath=use_fast_math_gpu)(base_compute_coeffs_vx)
    gpu_compute_coeffs_vy = cuda.jit(cache=True,
                                     inline="always",
                                     device=True,
                                     fastmath=use_fast_math_gpu)(base_compute_coeffs_vy)
    gpu_compute_neighbor_sum_vx = cuda.jit(cache=True,
                                           inline="always",
                                           device=True,
                                           fastmath=use_fast_math_gpu)(base_compute_neighbor_sum_vx)
    gpu_compute_neighbor_sum_vy = cuda.jit(cache=True,
                                           inline="always",
                                           device=True,
                                           fastmath=use_fast_math_gpu)(base_compute_neighbor_sum_vy)


    @cuda.jit(cache=True, inline="always", device=True, fastmath=use_fast_math_gpu)
    def gpu_inline_loop_body_vx(i: int, j: int,
                                dx: float, dy: float, relax_v: float,
                                etap: np.ndarray, etab: np.ndarray,
                                vx: np.ndarray, vy: np.ndarray, rhs: np.ndarray) -> float:
        """
        Contains full loop body for vx pass.
        """
        vx_c1, vx_c2, vx_c3, vx_c4, vx_c5, vy_c1, vy_c2, vy_c3, vy_c4 = gpu_compute_coeffs_vx(i=i, j=j,
                                                                                              dx=dx, dy=dy,
                                                                                              etap=etap, etab=etab)

        return gpu_compute_neighbor_sum_vx(i=i, j=j, relax_v=relax_v,
                                           vx=vx, vy=vy, rhs=rhs,
                                           vy_c3=vy_c3, vy_c2=vy_c2, vy_c4=vy_c4, vy_c1=vy_c1,
                                           vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4, vx_c5=vx_c5)

    @cuda.jit(cache=True, inline="always", device=True, fastmath=use_fast_math_gpu)
    def gpu_inline_loop_body_vy(i: int, j: int,
                                dx: float, dy: float, relax_v: float,
                                etap: np.ndarray, etab: np.ndarray,
                                vx: np.ndarray, vy: np.ndarray, rhs: np.ndarray) -> float:
        """
        Contains full loop body for vx pass.
        """
        vy_c1, vy_c2, vy_c3, vy_c4, vy_c5, vx_c1, vx_c2, vx_c3, vx_c4 = gpu_compute_coeffs_vy(i=i, j=j,
                                                                                              dx=dx, dy=dy,
                                                                                              etap=etap, etab=etab)

        return gpu_compute_neighbor_sum_vy(i=i, j=j, relax_v=relax_v,
                                           vx=vx, vy=vy, rhs=rhs,
                                           vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4,
                                           vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4, vy_c5=vy_c5)
else:
    def gpu_compute_coeffs_vx(*args, **kwargs):
        raise ImportError("numba-cuda needed for gpu support. Try installing Pyroclast[cuda]")

    def gpu_compute_coeffs_vy(*args, **kwargs):
        raise ImportError("numba-cuda needed for gpu support. Try installing Pyroclast[cuda]")

    def gpu_compute_neighbor_sum_vx(*args, **kwargs):
        raise ImportError("numba-cuda needed for gpu support. Try installing Pyroclast[cuda]")

    def gpu_compute_neighbor_sum_vy(*args, **kwargs):
        raise ImportError("numba-cuda needed for gpu support. Try installing Pyroclast[cuda]")

    def gpu_inline_loop_body_vx(*args, **kwargs):
        raise ImportError("numba-cuda needed for gpu support. Try installing Pyroclast[cuda]")

    def gpu_inline_loop_body_vy(*args, **kwargs):
        raise ImportError("numba-cuda needed for gpu support. Try installing Pyroclast[cuda]")
