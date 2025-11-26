"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/model/stokes_2D_mg/smoothers/rb_gs_fuse_cache_jitter.py
Description: Attempt 8: Reimplement the fully staggered loop with correct staggering. And Thread Blocking. And Boundary Jitter

Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


import numba as nb
import numpy as np
import  os
import traceback
from typing import Type
import math

from Pyroclast.solvers.stokes_2d.smoothers.inline_routines import cpu_inline_loop_body_vx, cpu_inline_loop_body_vy
from Pyroclast.solvers.stokes_2d.smoothers.rb_gs_fuse import velocity_smoother_rb_gs as base_rb_gs
from Pyroclast.solvers.stokes_2d.bc import cpu_apply_vx_BC, cpu_apply_vy_BC
from Pyroclast.utils import inject_threads


use_fast_math_cpu = os.environ.get("PYROCLAST_FASTMATH_CPU", default=False)


@inject_threads
@nb.njit(cache=True, parallel=True, fastmath=use_fast_math_cpu)
def velocity_smoother_rb_gs(nx1: int, ny1: int,
                            dx: float, dy: float,
                            etap: np.ndarray, etab: np.ndarray,
                            vx: np.ndarray, vy: np.ndarray,
                            relax_v: float, BC: float,
                            vx_rhs: np.ndarray, vy_rhs: np.ndarray, max_iter: int,
                            cache_a: int, step_size: int, th: int, max_jitter: int = None):
    if max_jitter is None:
        max_jitter = cache_a // 4

    # Fast Implementation for big problems
    if th * cache_a <= nx1 - 2:
        for _ in range(max_iter):
            jitter = np.random.randint(-max_jitter, max_jitter, th)

            # Constrain jitter of 0 to be 0
            jitter[0] = 0

            for p in nb.prange(th):
                start_x = p * (nx1 - 2) // th + 1 + jitter[p]
                end_x = (nx1 - 1) if p + 1 == th else (p + 1) * (nx1 - 2) // th + 1 + jitter[p + 1]

                blocks = math.ceil((end_x - start_x) / cache_a)

                # Iterate through the cache blocks
                for b in range(blocks):
                    start_bx = start_x + b * cache_a
                    end_bx = end_x if b + 1 == blocks else start_x + (b + 1) * cache_a

                    # Iterate through j, add + 3 for offset for second pass, and vy pass
                    for i in range(1, ny1 - 1 + step_size * 3):
                        for j in range(start_bx, end_bx):

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

    #
    else:
        raise ValueError("Problem too small for given resources")
        # base_rb_gs(nx1=nx1, ny1=ny1,
        #                         dx=dx, dy=dy,
        #                         etap=etap, etab=etab,
        #                         vx=vx, vy=vy,
        #                         relax_v=relax_v, BC=BC,
        #                         vx_rhs=vx_rhs, vy_rhs=vy_rhs, max_iter=max_iter, step_size=1)


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
        needs_cache_block_size_1: bool = True
        needs_jitter: bool = True

        def benchmark_preamble(self):
            start = dtf()
            ex_str = tb = None
            try:
                velocity_smoother_rb_gs(nx1=self.nx1, ny1=self.ny1, step_size=1,
                                        dx=self.dx, dy=self.dy,
                                        etap=self.eta_p, etab=self.eta_b,
                                        vx=self.vx, vy=self.vy,
                                        relax_v=self.relax_v, BC=self.boundary_condition,
                                        max_iter=1,
                                        vx_rhs=self.vx_rhs, vy_rhs=self.vy_rhs,
                                        cache_a=self.cache_block_size_1, max_jitter=self.jitter)
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
                velocity_smoother_rb_gs(nx1=self.nx1, ny1=self.ny1, step_size=1,
                                        dx=self.dx, dy=self.dy,
                                        etap=self.eta_p, etab=self.eta_b,
                                        vx=self.vx, vy=self.vy,
                                        relax_v=self.relax_v, BC=self.boundary_condition,
                                        max_iter=self.max_iter,
                                        vx_rhs=self.vx_rhs, vy_rhs=self.vy_rhs,
                                        cache_a=self.cache_block_size_1, max_jitter=self.jitter)
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

