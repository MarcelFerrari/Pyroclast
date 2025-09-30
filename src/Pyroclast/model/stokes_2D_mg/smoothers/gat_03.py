import math
import os
from typing import Type

import numba.cuda as cuda
import numpy as np
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from Pyroclast.model.stokes_2D_mg.smoothers.vx import gpu_inline_loop_body_vx
from Pyroclast.model.stokes_2D_mg.smoothers.vy import gpu_inline_loop_body_vy
from Pyroclast.model.stokes_2D_mg.utils import gpu_apply_vx_bc_kernel, gpu_apply_vy_bc_kernel


"""
GPU Implementation of the Smoother
-> Red-Black Gauss-Seidel with "fused operation" i.e. a single kernel that operates on different rows. 
-> No blocking or shifting of boundaries
-> interpolate from cuda grid straight to array - no translation or internal loops
-> Initial copy at the beginning and end of algorithm.
-> High likelyhood this is instable and will explode
"""

@cuda.jit
def rb_gs_sketchy_fuse(vx: DeviceNDArray, vy: DeviceNDArray,
                       etap: DeviceNDArray, etab: DeviceNDArray, vx_rhs: DeviceNDArray, vy_rhs: DeviceNDArray,
                       dx: float, dy: float, relax_v: float, nx1: int, ny1: int, step_size: int):
    """
    Cuda Kernel for Red Pass in VX
    """
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    iloc0 = i
    if (iloc0 + j) % 2 == 0 and 1 <= j <= nx1 - 2 and 1 <= iloc0 <= ny1 - 1:
        vx[iloc0, j] = gpu_inline_loop_body_vx(i=iloc0, j=j, dx=dx, dy=dy, relax_v=relax_v,
                                               etap=etap, etab=etab,
                                               vx=vx, vy=vy, rhs=vx_rhs)

    # Black Pass vx
    iloc1 = i - step_size
    if (iloc1 + j) % 2 == 1 and 1 <= j <= nx1 - 2 and 1 <= iloc1 <= ny1 - 1:
        vx[iloc1, j] = gpu_inline_loop_body_vx(i=iloc1, j=j, dx=dx, dy=dy, relax_v=relax_v,
                                               etap=etap, etab=etab,
                                               vx=vx, vy=vy, rhs=vx_rhs)

    # Red Pass vy
    iloc2 = i - (step_size * 2)
    if (iloc2 + j) % 2 == 0 and 1 <= j <= nx1 - 1 and 1 <= iloc2 <= ny1 - 2:
        vy[iloc2, j] = gpu_inline_loop_body_vy(i=iloc2, j=j, dx=dx, dy=dy, relax_v=relax_v,
                                               etap=etap, etab=etab,
                                               vx=vx, vy=vy, rhs=vy_rhs)

    # Black Pass vy
    iloc3 = i - (step_size * 3)
    if (iloc3 + j) % 2 == 0 and 1 <= j <= nx1 - 1 and 1 <= iloc3 <= ny1 - 2:
        vy[iloc3, j] = gpu_inline_loop_body_vy(i=iloc3, j=j, dx=dx, dy=dy, relax_v=relax_v,
                                               etap=etap, etab=etab,
                                               vx=vx, vy=vy, rhs=vy_rhs)


def setup_gpu(etap: np.ndarray, etab: np.ndarray,
              vx: np.ndarray, vy: np.ndarray,
              vx_rhs: np.ndarray, vy_rhs: np.ndarray) -> tuple[
    DeviceNDArray, DeviceNDArray, DeviceNDArray, DeviceNDArray,
    DeviceNDArray, DeviceNDArray]:
    """
    Allocate and copy arrays to GPU, returns all device arrays.
    """
    etap_d = cuda.to_device(etap)
    etab_d = cuda.to_device(etab)

    vx_d = cuda.to_device(vx)
    vy_d = cuda.to_device(vy)

    vx_rhs_d = cuda.to_device(vx_rhs)
    vy_rhs_d = cuda.to_device(vy_rhs)

    return vx_d, vy_d, etap_d, etab_d, vx_rhs_d, vy_rhs_d


# INFO: Cannot decorate with njit or jit. Doesn't work.
def velocity_smoother_rb_gs_cuda(
        nx1: int, ny1: int, max_iter: int,
        dx: float, dy: float, relax_v: float, BC: float,
        vx_d: DeviceNDArray, vy_d: DeviceNDArray,
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
        rb_gs_sketchy_fuse[grid_dim, threadsperblock](
            vx_d, vy_d,
            etap_d, etab_d, vx_rhs_d, vy_rhs_d,
            dx, dy, relax_v, nx1, ny1, 1)


        # INFO: This call should be technically unnecessary
        # cuda.synchronize()
        gpu_apply_vx_bc_kernel[grid_dim, threadsperblock](vx_d, BC, nx1, ny1)
        gpu_apply_vy_bc_kernel[grid_dim, threadsperblock](vy_d, BC, nx1, ny1)
        cuda.synchronize()

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
            # 0   1     2       3       4         5
            vx_d, vy_d, etap_d, etab_d, vx_rhs_d, vy_rhs_d = setup_gpu(
                etab=self.eta_b, etap=self.eta_p,
                vx=self.vx, vy=self.vy, vx_rhs=self.vx_rhs, vy_rhs=self.vy_rhs
            )
            self.device_arrays = [vx_d, vy_d, etap_d, etab_d, vx_rhs_d, vy_rhs_d]
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Copy to Device",
                                       stage=Stage.COPY_TO_DEVICE,
                                       start=start,
                                       end=end))

            start = dtf()
            velocity_smoother_rb_gs_cuda(nx1=self.nx1, ny1=self.ny1,
                                         dx=self.dx, dy=self.dy,
                                         etap_d=self.device_arrays[2], etab_d=self.device_arrays[3],
                                         vx_d=self.device_arrays[0], vy_d=self.device_arrays[1],
                                         relax_v=self.relax_v, BC=self.boundary_condition,
                                         max_iter=1,
                                         vx_rhs_d=self.device_arrays[4], vy_rhs_d=self.device_arrays[5])
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
            velocity_smoother_rb_gs_cuda(nx1=self.nx1, ny1=self.ny1,
                                         dx=self.dx, dy=self.dy,
                                         etap_d=self.device_arrays[2], etab_d=self.device_arrays[3],
                                         vx_d=self.device_arrays[0], vy_d=self.device_arrays[1],
                                         relax_v=self.relax_v, BC=self.boundary_condition,
                                         max_iter=1,
                                         vx_rhs_d=self.device_arrays[4], vy_rhs_d=self.device_arrays[5])
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Benchmark",
                                       stage=Stage.BENCHMARK,
                                       start=start,
                                       end=end))

    # INFO: Methods can be benchmarked in other places. Here is only the full implementation.
    return BaseImplementationBenchmarkSmoother, None, None