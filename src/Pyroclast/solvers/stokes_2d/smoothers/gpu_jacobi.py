"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/model/stokes_2D_mg/smoothers/gpu_jacobi.py
Description: GPU Attempt 1: Jacobi Smoother with gpu kernel

Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


import math
import os
from typing import Type

import numba.cuda as cuda
import numpy as np
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from Pyroclast.solvers.stokes_2d.smoothers.inline_routines import gpu_inline_loop_body_vx, gpu_inline_loop_body_vy
from Pyroclast.solvers.stokes_2d.bc import gpu_apply_vx_bc_kernel, gpu_apply_vy_bc_kernel


use_fast_math_gpu = os.environ.get("PYROCLAST_FASTMATH_GPU", default=False)
IS_GPU = True


"""
GPU Implementation of the Smoother
-> Basic Jacobi.
-> No blocking or shifting of boundaries
-> interpolate from cuda grid straight to array - no translation or internal loops
-> Initial copy at the beginning and end of algorithm.
"""


@cuda.jit(device=True, cache=True, fastmath=use_fast_math_gpu)
def jacobi_step_kernel(vx: DeviceNDArray, vy: DeviceNDArray, vx_new: DeviceNDArray, vy_new: DeviceNDArray,
                       etap: DeviceNDArray, etab: DeviceNDArray, vx_rhs: DeviceNDArray, vy_rhs: DeviceNDArray,
                       dx: float, dy: float, relax_v: float, nx1: int, ny1: int):
    """
    Cuda Kernel to perform Jacobi step
    """
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if 1 <= j <= nx1 - 2 and 1 <= i <= ny1 - 1:
        vx_new[i, j] = gpu_inline_loop_body_vx(
            i, j, dx, dy, relax_v, etap, etab, vx, vy, vx_rhs)

    if 1 <= j <= nx1 - 1 and 1 <= i <= ny1 - 2:
        vy_new[i, j] = gpu_inline_loop_body_vy(
            i, j, dx, dy, relax_v, etap, etab, vx, vy, vy_rhs)


def setup_gpu(etap: np.ndarray, etab: np.ndarray,
              vx: np.ndarray, vy: np.ndarray,
              vx_rhs: np.ndarray, vy_rhs: np.ndarray) -> tuple[
    DeviceNDArray, DeviceNDArray, DeviceNDArray, DeviceNDArray,
    DeviceNDArray, DeviceNDArray, DeviceNDArray, DeviceNDArray]:
    """
    Allocate and copy arrays to GPU, returns all device arrays.
    """
    etap_d = cuda.to_device(etap)
    etab_d = cuda.to_device(etab)

    vx_d = cuda.to_device(vx)
    vy_d = cuda.to_device(vy)

    vx_new_d = cuda.device_array_like(vx_d)
    vy_new_d = cuda.device_array_like(vy_d)

    vx_rhs_d = cuda.to_device(vx_rhs)
    vy_rhs_d = cuda.to_device(vy_rhs)

    return vx_d, vy_d, vx_new_d, vy_new_d, etap_d, etab_d, vx_rhs_d, vy_rhs_d


# INFO: Cannot decorate with njit or jit. Doesn't work. Also @cuda.jit doesn't work. Expects to be a kernel
def velocity_smoother_jacobi_cuda(
        nx1: int, ny1: int, max_iter: int,
        dx: float, dy: float, relax_v: float, BC: float,
        vx_d: DeviceNDArray, vy_d: DeviceNDArray, vx_new_d: DeviceNDArray, vy_new_d: DeviceNDArray,
        etap_d: DeviceNDArray, etab_d: DeviceNDArray, vx_rhs_d: DeviceNDArray, vy_rhs_d: DeviceNDArray,
        threadsperblock: tuple[int, int] = (32, 32)):
    """
    Run Jacobi smoother on GPU with provided device arrays.
    Returns vx, vy copied back to host.
    """
    bx, by = threadsperblock
    blockspergrid_x = math.ceil(nx1 / bx)
    blockspergrid_y = math.ceil(ny1 / by)
    grid_dim = (blockspergrid_x, blockspergrid_y)

    for it in range(max_iter):
        jacobi_step_kernel[grid_dim, threadsperblock](
            vx_d, vy_d, vx_new_d, vy_new_d,
            etap_d, etab_d, vx_rhs_d, vy_rhs_d,
            dx, dy, relax_v, nx1, ny1)

        # INFO: This call should be technically unnecessary
        # cuda.synchronize()
        gpu_apply_vx_bc_kernel[grid_dim, threadsperblock](vx_new_d, BC, nx1, ny1)
        gpu_apply_vy_bc_kernel[grid_dim, threadsperblock](vy_new_d, BC, nx1, ny1)
        cuda.synchronize()

        vx_d, vx_new_d = vx_new_d, vx_d
        vy_d, vy_new_d = vy_new_d, vy_d

    return vx_d, vy_d


# INFO need to use string references to avoid circular imports and deal with benchmark packaged not available
def benchmark_factory() -> tuple[Type["BenchmarkSmoother"], Type["BenchmarkVX"], Type["BenchmarkVY"]]:
    """
    Returns Benchmark Classes needed for benchmarking. Done via factory to avoid issues with the `benchmark` package
    not being available in a production environment.
    """
    import benchmark.benchmark_wrapper as bw
    from benchmark.benchmark_validators import Stage, Timing
    from benchmark.utils import dtf

    module_name = os.path.basename(__file__).replace(".py", "")

    class BaseImplementationBenchmarkSmoother(bw.BenchmarkSmoother):
        device_arrays: list[DeviceNDArray] = []

        def benchmark_preamble(self):
            start = dtf()
            # 0   1     2        3          4       5       6        7
            vx_d, vy_d, vx_new_d, vy_new_d, etap_d, etab_d, vx_rhs_d, vy_rhs_d = setup_gpu(
                etab=self.eta_b, etap=self.eta_p,
                vx=self.vx, vy=self.vy, vx_rhs=self.vx_rhs, vy_rhs=self.vy_rhs
            )
            self.device_arrays = [vx_d, vy_d, vx_new_d, vy_new_d, etap_d, etab_d, vx_rhs_d, vy_rhs_d]
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Copy to Device",
                                       stage=Stage.COPY_TO_DEVICE,
                                       start=start,
                                       end=end))

            start = dtf()
            velocity_smoother_jacobi_cuda(nx1=self.nx1, ny1=self.ny1,
                                          dx=self.dx, dy=self.dy,
                                          etap_d=self.device_arrays[4], etab_d=self.device_arrays[5],
                                          vx_d=self.device_arrays[0], vy_d=self.device_arrays[1],
                                          vx_new_d=self.device_arrays[2], vy_new_d=self.device_arrays[3],
                                          relax_v=self.relax_v, BC=self.boundary_condition,
                                          max_iter=1,
                                          vx_rhs_d=self.device_arrays[6], vy_rhs_d=self.device_arrays[7])
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Preamble - Compile",
                                       stage=Stage.PREAMBLE,
                                       start=start,
                                       end=end))

        def benchmark_epilogue(self):
            """
            No post-processing
            """
            pass

        def run_benchmark(self):
            """
            Perform the actual run of the benchmark.
            """
            start = dtf()
            velocity_smoother_jacobi_cuda(nx1=self.nx1, ny1=self.ny1,
                                          dx=self.dx, dy=self.dy,
                                          etap_d=self.device_arrays[4], etab_d=self.device_arrays[5],
                                          vx_d=self.device_arrays[0], vy_d=self.device_arrays[1],
                                          vx_new_d=self.device_arrays[2], vy_new_d=self.device_arrays[3],
                                          relax_v=self.relax_v, BC=self.boundary_condition,
                                          max_iter=1,
                                          vx_rhs_d=self.device_arrays[6], vy_rhs_d=self.device_arrays[7])
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Benchmark",
                                       stage=Stage.BENCHMARK,
                                       start=start,
                                       end=end))

    # INFO: Methods can be benchmarked in other places. Here is only the full implementation.
    return BaseImplementationBenchmarkSmoother, None, None
