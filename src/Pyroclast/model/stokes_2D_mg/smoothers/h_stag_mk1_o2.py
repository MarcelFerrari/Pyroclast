"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/model/stokes_2D_mg/smoothers/h_stag_mk1_o2.py 
Description: Internal If implementation of staggered red-black gauss-seidel, offset is 2, and half staggered 
             implementation, using cache optimized blocking

Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


import numba as nb
import numpy as np
import  os
from typing import Type
import math

from Pyroclast.model.stokes_2D_mg.smoothers.vx.inline_vx import inline_loop_body_vx
from Pyroclast.model.stokes_2D_mg.smoothers.vy.inline_vy import inline_loop_body_vy
from Pyroclast.model.stokes_2D_mg.utils import apply_vx_BC, apply_vy_BC


@nb.njit(cache=True, parallel=True)
def velocity_smoother_rb_gs(nx1: int, ny1: int,
                            dx: float, dy: float,
                            etap: np.ndarray, etab: np.ndarray,
                            vx: np.ndarray, vy: np.ndarray,
                            relax_v: float, BC: float,
                            vx_rhs: np.ndarray, vy_rhs: np.ndarray, max_iter: int,
                            th: int, cache_a: int):
    for _ in range(max_iter):
        # Work Split for red pass
        for p in nb.prange(th):
            start_y = p * (ny1 - 2) / th + 1
            end_y = (ny1 - 1) if p + 1 == th else (p + 1) * (ny1 - 2) / th + 1

            blocks = math.ceil((end_y - start_y) / cache_a)

            # Iterate through the cache blocks
            for b in range(blocks):
                start_b = start_y + b * cache_a
                end_b = end_y if b + 1 == blocks else start_y + (b + 1) * cache_a

                # Iterate through j, add + 2 for offset for second pass
                for j in range(1, nx1 - 2 + 2):
                    for i in range(start_b, end_b):
                        # Red pass
                        if 1 <= j < nx1 - 2 and (i + j) % 2 == 0:
                            # Red pass vx, vy
                            vx[i, j] = inline_loop_body_vx(i=i, j=j, dx=dx, dy=dy, relax_v=relax_v,
                                                           etap=etap, etab=etab,
                                                           vx=vx, vy=vy, rhs=vx_rhs)

                            vy[i, j] = inline_loop_body_vy(i=i, j=j, dx=dx, dy=dy, relax_v=relax_v,
                                                           etap=etap, etab=etab,
                                                           vx=vx, vy=vy, rhs=vy_rhs)

                        # Black pass
                        if 1 <= j-2 < nx1 - 2 and (i + j - 2) % 2 == 1:
                            # Black pass vx, vy
                            vx[i, j - 2] = inline_loop_body_vx(i=i, j=j - 2, dx=dx, dy=dy, relax_v=relax_v,
                                                               etap=etap, etab=etab,
                                                               vx=vx, vy=vy, rhs=vx_rhs)

                            vy[i, j - 2] = inline_loop_body_vy(i=i, j=j - 2, dx=dx, dy=dy, relax_v=relax_v,
                                                               etap=etap, etab=etab,
                                                               vx=vx, vy=vy, rhs=vy_rhs)

        apply_vx_BC(vx, BC)
        apply_vy_BC(vy, BC)


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

    # TODO finish benchmark
    class BaseImplementationBenchmarkSmoother(bw.BenchmarkSmoother):
        def benchmark_preamble(self):
            start = dtf()
            velocity_smoother_rb_gs(nx1=self.nx1, ny1=self.ny1,
                                    dx=self.dx, dy=self.dy,
                                    etap=self.eta_p, etab=self.eta_b,
                                    vx=self.vx, vy=self.vy,
                                    relax_v=self.relax_v, BC=self.boundary_condition,
                                    max_iter=1,
                                    vx_rhs=self.vx_rhs, vy_rhs=self.vy_rhs)
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Preamble",
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
            velocity_smoother_rb_gs(nx1=self.nx1, ny1=self.ny1,
                                    dx=self.dx, dy=self.dy,
                                    etap=self.eta_p, etab=self.eta_b,
                                    vx=self.vx, vy=self.vy,
                                    relax_v=self.relax_v, BC=self.boundary_condition,
                                    max_iter=self.max_iter,
                                    vx_rhs=self.vx_rhs, vy_rhs=self.vy_rhs)
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Benchmark",
                                       stage=Stage.BENCHMARK,
                                       start=start,
                                       end=end))

    # INFO: Methods can be benchmarked in other places. Here is only the full implementation.
    return BaseImplementationBenchmarkSmoother, None, None
