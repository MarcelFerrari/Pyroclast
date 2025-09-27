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
from ._inline_vy import cpu_inline_loop_body_vy


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
            vy[i, j] = cpu_inline_loop_body_vy(i=i, j=j, relax_v=relax_v,
                                               vx=vx, vy=vy, rhs=rhs,
                                               dx=dx, dy=dy,
                                               etab=etab, etap=etap)

        i = 1 + 1
        j_start = 1 if i % 2 == 0 else 2  # Red pass starts on even (i+j)
        for j in range(j_start, nx1 - 2, 2):
            vy[i, j] = cpu_inline_loop_body_vy(i=i, j=j, relax_v=relax_v,
                                               vx=vx, vy=vy, rhs=rhs,
                                               dx=dx, dy=dy,
                                               etab=etab, etap=etap)
    else:
        for p in nb.prange(th):
            start = p * (nx1 - 3) / th
            end = (nx1 - 2) if p + 1 == th else (p + 1) * (nx1 - 3) / th

            for i in range(start, end):
                for j in range(1, 3):

                    # Abort if we have odd pairing
                    if i + j != 0:
                        continue

                    vy[i, j] = cpu_inline_loop_body_vy(i=i, j=j, relax_v=relax_v,
                                                       vx=vx, vy=vy, rhs=rhs,
                                                       dx=dx, dy=dy,
                                                       etab=etab, etap=etap)

    for i in nb.prange(1 + 2, ny1 - 1):
        # assert 1 <= i < ny1 -1, "Error with bounds - red pass"
        j_start = 1 if i % 2 == 0 else 2  # Red pass starts on even (i+j)
        for j in range(j_start, nx1 - 2, 2):
            vy[i, j] = cpu_inline_loop_body_vy(i=i, j=j, relax_v=relax_v,
                                               vx=vx, vy=vy, rhs=rhs,
                                               dx=dx, dy=dy,
                                               etab=etab, etap=etap)
        # assert 1 <= i - 2 < ny1 -1, "Error with bounds - black pass"
        j_start = 2 if (i - 2) % 2 == 0 else 1  # Black pass starts on odd (i+j)
        for j in range(j_start, nx1 - 2, 2):
            vy[i, j] = cpu_inline_loop_body_vy(i=i, j=j, relax_v=relax_v,
                                               vx=vx, vy=vy, rhs=rhs,
                                               dx=dx, dy=dy,
                                               etab=etab, etap=etap)

    if th < (nx1 - 4) / 2:
        # Tear down pipeline with last two rows of the black pass
        i = nx1 - 1 - 2
        j_start = 2 if (i - 2) % 2 == 0 else 1  # Black pass starts on odd (i+j)
        for j in range(j_start, nx1 - 2, 2):
            vy[i, j] = cpu_inline_loop_body_vy(i=i, j=j, relax_v=relax_v,
                                               vx=vx, vy=vy, rhs=rhs,
                                               dx=dx, dy=dy,
                                               etab=etab, etap=etap)

        i = nx1 - 1 - 1
        j_start = 2 if (i - 2) % 2 == 0 else 1  # Black pass starts on odd (i+j)
        for j in range(j_start, nx1 - 2, 2):
            vy[i, j] = cpu_inline_loop_body_vy(i=i, j=j, relax_v=relax_v,
                                               vx=vx, vy=vy, rhs=rhs,
                                               dx=dx, dy=dy,
                                               etab=etab, etap=etap)
    else:
        for p in nb.prange(th):
            start = p * (nx1 - 3) / th
            end = (nx1 - 2) if p + 1 == th else (p + 1) * (nx1 - 3) / th

            for i in range(start, end):
                for j in range(1, 3):

                    # Abort if we have odd pairing
                    if i + j == 0:
                        continue

                    vy[i, j] = cpu_inline_loop_body_vy(i=i, j=j, relax_v=relax_v,
                                                       vx=vx, vy=vy, rhs=rhs,
                                                       dx=dx, dy=dy,
                                                       etab=etab, etap=etap)

    apply_vy_BC(vy, BC)

    return vy


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

    class BaseImplementationVY(bw.BenchmarkVY):
        def benchmark_preamble(self):
            start = dtf()
            _vy_rb_gs_sweep(nx1=self.nx1, ny1=self.ny1,
                            dx=self.dx, dy=self.dy,
                            etap=self.eta_p, etab=self.eta_b,
                            vx=self.vx, vy=self.vy,
                            relax_v=self.relax_v, BC=self.boundary_condition, rhs=self.vy_rhs,
                            th=nb.get_num_threads())
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
            for _ in range(self.args.max_iter):
                _vy_rb_gs_sweep(nx1=self.nx1, ny1=self.ny1,
                                dx=self.dx, dy=self.dy,
                                etap=self.eta_p, etab=self.eta_b,
                                vx=self.vx, vy=self.vy,
                                relax_v=self.relax_v, BC=self.boundary_condition, rhs=self.vy_rhs,
                                th=nb.get_num_threads())
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Benchmark",
                                       stage=Stage.BENCHMARK,
                                       start=start,
                                       end=end))

    return None, None, BaseImplementationVY