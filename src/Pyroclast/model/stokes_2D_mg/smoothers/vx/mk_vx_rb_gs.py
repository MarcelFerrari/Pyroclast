"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/model/stokes_2D_mg/smoothers/vx/mk_vx_rb_gs.py
Description: In this file we test both a cache optimized version of the red_black gauss_seidel sweep with a staggered 
             computation of the red and black stage using an iff

Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


import math
import os.path
from typing import Type

import numba as nb
import numpy as np

from Pyroclast.model.stokes_2D_mg.utils import apply_vx_BC
from ._inline_vx import cpu_compute_coeffs_vx, cpu_compute_neighbor_sum_vx


# Cache optimized version of the regular red-black gauss-seidel

@nb.njit(cache=True, parallel=True)
def _vx_rb_gs_sweep(nx1, ny1,
                    dx, dy,
                    etap, etab,
                    vx, vy,
                    relax_v, rhs, BC,
                    th: int, cache_a: int) -> np.ndarray:
    """
    In-place Red-Black Gauss-Seidel update for vx.
    """
    # Case when we less cores than we want
    if th > (ny1 - 2):
        # ----------------------------
        #  Red pass: (i + j) % 2 == 0
        # ----------------------------
        for i in nb.prange(1, ny1 - 1):
            j_start = 1 if i % 2 == 0 else 2  # Red pass starts on even (i+j)
            for j in range(j_start, nx1 - 2, 2):
                vx_c1, vx_c2, vx_c3, vx_c4, vx_c5, vy_c1, vy_c2, vy_c3, vy_c4 = cpu_compute_coeffs_vx(i, j, dx, dy, etap, etab)

                # Gauss-Seidel in-place update
                vx[i, j] = cpu_compute_neighbor_sum_vx(
                    i, j, relax_v, vx_c1, vx_c2, vx_c3, vx_c4, vx_c5, vy_c1, vy_c2, vy_c3, vy_c4, vx, vy, rhs
                )

        # Apply vx boundary conditions
        apply_vx_BC(vx, BC)

        # ----------------------------
        #  Black pass: (i + j) % 2 == 1
        # ----------------------------
        for i in nb.prange(1, ny1 - 1):
            j_start = 2 if i % 2 == 0 else 1  # Black pass starts on odd (i+j)
            for j in range(j_start, nx1 - 2, 2):
                vx_c1, vx_c2, vx_c3, vx_c4, vx_c5, vy_c1, vy_c2, vy_c3, vy_c4 = cpu_compute_coeffs_vx(i, j, dx, dy, etap, etab)

                # Gauss-Seidel in-place update
                vx[i, j] = cpu_compute_neighbor_sum_vx(
                    i, j, relax_v, vx_c1, vx_c2, vx_c3, vx_c4, vx_c5, vy_c1, vy_c2, vy_c3, vy_c4, vx, vy, rhs
                )

    else:
        # Work Split for red pass
        for p in nb.prange(th):
            start_y = p * (ny1 - 2) / th + 1
            end_y = (ny1 - 1) if p + 1 == th else (p + 1) * (ny1 - 2) / th + 1

            blocks = math.ceil((end_y - start_y) / cache_a)

            # Iterate through the cache blocks
            for b in range(blocks):
                start_b = start_y + b * cache_a
                end_b = end_y if b + 1 == blocks else start_y + (b + 1) * cache_a

                # Iterate through j
                for j in range(1, nx1 - 2):
                    for i in range(start_b, end_b):
                        if (i + j) % 2 == 1:
                            continue

                        vx_c1, vx_c2, vx_c3, vx_c4, vx_c5, vy_c1, vy_c2, vy_c3, vy_c4 = cpu_compute_coeffs_vx(
                            i, j, dx, dy, etap, etab)

                        # Gauss-Seidel in-place update
                        vx[i, j] = cpu_compute_neighbor_sum_vx(
                            i, j, relax_v, vx_c1, vx_c2, vx_c3, vx_c4, vx_c5, vy_c1, vy_c2, vy_c3, vy_c4, vx, vy, rhs
                        )

        # INFO: Join needed for race conditions

        # Work Split for red pass
        for p in nb.prange(th):
            start_y = p * (ny1 - 2) / th + 1
            end_y = (ny1 - 1) if p + 1 == th else (p + 1) * (ny1 - 2) / th + 1

            blocks = math.ceil((end_y - start_y) / cache_a)

            # Iterate through the cache blocks
            for b in range(blocks):
                start_b = start_y + b * cache_a
                end_b = end_y if b + 1 == blocks else start_y + (b + 1) * cache_a

                # Iterate through j
                for j in range(1, nx1 - 2):
                    for i in  range(start_b, end_b):
                        if (i + j) % 2 == 0:
                            continue

                        vx_c1, vx_c2, vx_c3, vx_c4, vx_c5, vy_c1, vy_c2, vy_c3, vy_c4 = cpu_compute_coeffs_vx(
                            i, j, dx, dy, etap, etab)

                        # Gauss-Seidel in-place update
                        vx[i, j] = cpu_compute_neighbor_sum_vx(
                            i, j, relax_v, vx_c1, vx_c2, vx_c3, vx_c4, vx_c5, vy_c1, vy_c2, vy_c3, vy_c4, vx, vy, rhs
                        )

    # Apply vx boundary conditions
    apply_vx_BC(vx, BC)

    return vx


# INFO need to use string references to avoid circular imports and deal with benchmark packaged not available
def benchmark_factory() -> tuple[Type["BenchmarkSmoother"], Type["BenchmarkVX"], Type["BenchmarkVY"]]:
    """
    Returns Benchmark Classes needed for benchmarking. Done via factory to avoid issues with the `benchmark` package
    not being available in a production environment.
    """
    import benchmark.benchmark_wrapper as bw
    from benchmark.benchmark_validators import Stage, Timing, BenchmarkValidatorVX
    from benchmark.utils import dtf

    module_name = os.path.basename(__file__).replace(".py", "")

    class BaseImplementationVX(bw.BenchmarkVX):
        needs_cache_block_size_1: bool = True

        def benchmark_preamble(self):
            th = nb.get_num_threads()
            start = dtf()
            _vx_rb_gs_sweep(nx1=self.nx1, ny1=self.ny1,
                            dx=self.dx, dy=self.dy,
                            etap=self.eta_p, etab=self.eta_b,
                            vx=self.vx, vy=self.vy,
                            relax_v=self.relax_v, BC=self.boundary_condition, rhs=self.vx_rhs,
                            th=th, cache_a=self.cache_block_size_1)
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
            th = nb.get_num_threads()
            start = dtf()
            for _ in range(self.args.max_iter):
                _vx_rb_gs_sweep(nx1=self.nx1, ny1=self.ny1,
                                dx=self.dx, dy=self.dy,
                                etap=self.eta_p, etab=self.eta_b,
                                vx=self.vx, vy=self.vy,
                                relax_v=self.relax_v, BC=self.boundary_condition, rhs=self.vx_rhs,
                                th=th, cache_a=self.cache_block_size_1)
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Benchmark",
                                       stage=Stage.BENCHMARK,
                                       start=start,
                                       end=end))

    return None, BaseImplementationVX, None
