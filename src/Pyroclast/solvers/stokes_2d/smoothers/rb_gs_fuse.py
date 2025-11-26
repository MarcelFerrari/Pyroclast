"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/model/stokes_2D_mg/smoothers/rb_gs_fuse.py 
Description: Attempt 12: Reimplement the fully staggered loop with correct staggering. No Thread blocking for smaller problems.
    
Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import os
from typing import Type
import traceback

import numba as nb
import numpy as np

from Pyroclast.solvers.stokes_2d.smoothers.inline_routines import cpu_inline_loop_body_vx, cpu_inline_loop_body_vy
from Pyroclast.solvers.stokes_2d.bc import cpu_apply_vx_BC, cpu_apply_vy_BC


use_fast_math_cpu = os.environ.get("PYROCLAST_FASTMATH_CPU", default=False)


@nb.njit(cache=True, parallel=True, fastmath=use_fast_math_cpu)
def velocity_smoother_rb_gs(nx1: int, ny1: int,
                            dx: float, dy: float,
                            etap: np.ndarray, etab: np.ndarray,
                            vx: np.ndarray, vy: np.ndarray,
                            relax_v: float, BC: float,
                            vx_rhs: np.ndarray, vy_rhs: np.ndarray, max_iter: int,
                            step_size: int):
    for _ in range(max_iter):
        for i in nb.prange(1, ny1 - 1 + step_size * 3):
            for j in range(1 , nx1 - 1):

                # Red Pass vx
                iloc0 = i
                if (iloc0 + j) % 2 == 0 and 1 <= j <= nx1 - 2 and 1 <= iloc0 <= ny1 - 1:
                    vx[iloc0, j] = cpu_inline_loop_body_vx(i=iloc0, j=j, dx=dx, dy=dy, relax_v=relax_v,
                                                           etap=etap, etab=etab,
                                                           vx=vx, vy=vy, rhs=vx_rhs)

                # Black Pass vx
                iloc1 = i - step_size
                if (iloc1 + j) % 2 == 1 and 1 <= j <= nx1 - 2 and 1 <= iloc1 <= ny1 - 1:
                    vx[iloc1, j] = cpu_inline_loop_body_vx(i=iloc1, j=j, dx=dx, dy=dy, relax_v=relax_v,
                                                           etap=etap, etab=etab,
                                                           vx=vx, vy=vy, rhs=vx_rhs)

                # Red Pass vy
                iloc2 = i - (step_size * 2)
                if (iloc2 + j) % 2 == 0 and 1 <= j <= nx1 - 1 and 1 <= iloc2 <= ny1 - 2:
                        vy[iloc2, j] = cpu_inline_loop_body_vy(i=iloc2, j=j, dx=dx, dy=dy, relax_v=relax_v,
                                                               etap=etap, etab=etab,
                                                               vx=vx, vy=vy, rhs=vy_rhs)

                # Black Pass vy
                iloc3 = i - (step_size * 3)
                if (iloc3 + j) % 2 == 0 and 1 <= j <= nx1 - 1 and 1 <= iloc3 <= ny1 - 2:
                        vy[iloc3, j] = cpu_inline_loop_body_vy(i=iloc3, j=j, dx=dx, dy=dy, relax_v=relax_v,
                                                               etap=etap, etab=etab,
                                                               vx=vx, vy=vy, rhs=vy_rhs)


        cpu_apply_vx_BC(vx, BC)
        cpu_apply_vy_BC(vy, BC)


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
        def benchmark_preamble(self):
            start = dtf()
            ex_str = tb = None
            try:
                velocity_smoother_rb_gs(nx1=self.nx1, ny1=self.ny1,
                                        dx=self.dx, dy=self.dy,
                                        etap=self.eta_p, etab=self.eta_b,
                                        vx=self.vx, vy=self.vy,
                                        relax_v=self.relax_v, BC=self.boundary_condition,
                                        max_iter=1,
                                        vx_rhs=self.vx_rhs, vy_rhs=self.vy_rhs, step_size=1)
            except Exception as e:
                ex_str = str(e)
                tb = traceback.format_exc()
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
                velocity_smoother_rb_gs(nx1=self.nx1, ny1=self.ny1,
                                        dx=self.dx, dy=self.dy,
                                        etap=self.eta_p, etab=self.eta_b,
                                        vx=self.vx, vy=self.vy,
                                        relax_v=self.relax_v, BC=self.boundary_condition,
                                        max_iter=self.max_iter,
                                        vx_rhs=self.vx_rhs, vy_rhs=self.vy_rhs, step_size=1)
            except Exception as e:
                ex_str = str(e)
                tb = traceback.format_exc()
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
