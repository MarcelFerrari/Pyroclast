"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/model/stokes_2D_mg/smoothers/jacobi_fuse.py 
Description: Attempt 5: Implemented Blocking Jacobi. And Thread Blocking

Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import os
import traceback
from typing import Type

import numba as nb
import numpy as np

from Pyroclast.solvers.stokes_2d.smoothers.inline_routines import cpu_inline_loop_body_vx, cpu_inline_loop_body_vy
from Pyroclast.solvers.stokes_2d.bc import cpu_apply_vx_BC, cpu_apply_vy_BC


use_fast_math_cpu = os.environ.get("PYROCLAST_FASTMATH_CPU", default=False)


@nb.njit(cache=True, parallel=True, fastmath=use_fast_math_cpu)
def velocity_smoother_jacobi(nx1: int, ny1: int,
                             dx: float, dy: float,
                             etap: np.ndarray, etab: np.ndarray,
                             vx: np.ndarray, vy: np.ndarray,
                             relax_v: float, BC: float,
                             vx_rhs: np.ndarray, vy_rhs: np.ndarray, max_iter: int,
                             vx_new: np.ndarray, vy_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Fast Implementation for big problems
    for _ in range(max_iter):
        # Iterate through j, add + 3 for offset for second pass, and vy pass
        for i in nb.prange(1, ny1 - 1):
            for j in range(1, nx1 - 1):
                # Pass vx
                if 1 <= j <= nx1 - 2 and 1 <= i <= ny1 - 1:
                    vx_new[i, j] = cpu_inline_loop_body_vx(i=i, j=j, dx=dx, dy=dy, relax_v=relax_v,
                                                           etap=etap, etab=etab,
                                                           vx=vx, vy=vy, rhs=vx_rhs)

                # Pass vy
                if 1 <= j <= nx1 - 1 and 1 <= i <= ny1 - 2:
                    vy_new[i, j] = cpu_inline_loop_body_vy(i=i, j=j, dx=dx, dy=dy, relax_v=relax_v,
                                                           etap=etap, etab=etab,
                                                           vx=vx, vy=vy, rhs=vy_rhs)

        cpu_apply_vx_BC(vx_new, BC)
        cpu_apply_vy_BC(vy_new, BC)
        vx, vx_new = vx_new, vx
        vy, vy_new = vy_new, vy

    return vx, vy




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
        even_iter: bool = True

        def __init__(self, arguments: bw.BenchmarkValidatorSmoother):
            super().__init__(arguments=arguments)

            if self.vx_new is None:
                self.vx_new = np.zeros((self.nx1, self.ny1))

            if self.vy_new is None:
                self.vy_new = np.zeros((self.nx1, self.ny1))

        def benchmark_preamble(self):
            start = dtf()
            ex_str = tb = None
            try:
                velocity_smoother_jacobi(nx1=self.nx1, ny1=self.ny1,
                                         dx=self.dx, dy=self.dy,
                                         etap=self.eta_p, etab=self.eta_b,
                                         vx=self.vx, vy=self.vy, vx_new=self.vx_new, vy_new=self.vy_new,
                                         relax_v=self.relax_v, BC=self.boundary_condition,
                                         max_iter=1,
                                         vx_rhs=self.vx_rhs, vy_rhs=self.vy_rhs)
            except Exception as e:
                ex_str = str(e)
                tb = traceback.format_exc()
                print(tb)

            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Preamble",
                                       stage=Stage.PREAMBLE,
                                       start=start,
                                       end=end,
                                       error=ex_str,
                                       traceback=tb))

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
            ex_str = tb = None
            try:
                velocity_smoother_jacobi(nx1=self.nx1, ny1=self.ny1,
                                         dx=self.dx, dy=self.dy,
                                         etap=self.eta_p, etab=self.eta_b,
                                         vx=self.vx, vy=self.vy, vx_new=self.vx_new, vy_new=self.vy_new,
                                         relax_v=self.relax_v, BC=self.boundary_condition,
                                         max_iter=self.max_iter,
                                         vx_rhs=self.vx_rhs, vy_rhs=self.vy_rhs)
            except Exception as e:
                ex_str = str(e)
                tb = traceback.format_exc()
                print(tb)

            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Benchmark",
                                       stage=Stage.BENCHMARK,
                                       start=start,
                                       end=end,
                                       error=ex_str,
                                       traceback=tb))

    # INFO: Methods can be benchmarked in other places. Here is only the full implementation.
    return BaseImplementationBenchmarkSmoother, None, None
