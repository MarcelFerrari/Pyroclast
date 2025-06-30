"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/model/stokes_2D_mg/smoothers/f_stag_v1.py 
Description: File contains fused loop velocity smoother. Pass distance is 2, NOT cache optimized.

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


import Pyroclast.model.stokes_2D_mg.smoothers.vx.inline_vx as vx_op
import Pyroclast.model.stokes_2D_mg.smoothers.vy.inline_vy as vy_op
from Pyroclast.model.stokes_2D_mg.utils import apply_vx_BC, apply_vy_BC


@nb.njit(cache=True, parallel=True)
def velocity_smoother_rb_gs(nx1: int, ny1: int,
                            dx: float, dy: float,
                            etap: np.ndarray, etab: np.ndarray,
                            vx: np.ndarray, vy: np.ndarray,
                            relax_v: float, BC: float,
                            vx_rhs: np.ndarray, vy_rhs: np.ndarray, max_iter: int):
    for _ in range(max_iter):
        for i in nb.prange(1, ny1 - 1 + 6):
            i_loc_1 = i
            if 1 <= i_loc_1 < ny1 - 1:
                j_start = 1 if i_loc_1 % 2 == 0 else 2  # Red pass starts on even (i+j)
                for j in range(j_start, nx1 - 2, 2):
                    vx_c1, vx_c2, vx_c3, vx_c4, vx_c5, vy_c1, vy_c2, vy_c3, vy_c4 = vx_op.compute_coeffs(
                        i=i_loc_1, j=j, dx=dx, dy=dy, etap=etap, etab=etab)

                    # Gauss-Seidel in-place update
                    vx[i_loc_1, j] = vx_op.compute_neighbor_sum(
                        i=i_loc_1, j=j, relax_v=relax_v,
                        vx=vx, vy=vy, rhs=vx_rhs,
                        vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4, vx_c5=vx_c5,
                        vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4
                    )

            # Staggered black pass
            i_loc_2 = i - 2
            if 1 <= i_loc_2 <= ny1 - 1:
                j_start = 2 if i_loc_2 % 2 == 0 else 1  # Black pass starts on odd (i+j)
                for j in range(j_start, nx1 - 2, 2):
                    vx_c1, vx_c2, vx_c3, vx_c4, vx_c5, vy_c1, vy_c2, vy_c3, vy_c4 = vx_op.compute_coeffs(
                        i=i_loc_2, j=j, dx=dx, dy=dy, etap=etap, etab=etab)

                    # Gauss-Seidel in-place update
                    vx[i_loc_2, j] = vx_op.compute_neighbor_sum(
                        i=i_loc_2, j=j, relax_v=relax_v,
                        vx=vx, vy=vy, rhs=vx_rhs,
                        vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4, vx_c5=vx_c5,
                        vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4
                    )

            i_loc_3 = i - 4
            if 1 <= i_loc_3 <= ny1 - 1:
                j_start = 1 if i_loc_3 % 2 == 0 else 2  # Red pass starts on even (i+j)
                for j in range(j_start, nx1 - 2, 2):
                    vy_c1, vy_c2, vy_c3, vy_c4, vy_c5, vx_c1, vx_c2, vx_c3, vx_c4 = vy_op.compute_coeffs(
                        i=i_loc_3, j=j, dx=dx, dy=dy, etap=etap, etab=etab)

                    vy[i_loc_3, j] = vy_op.compute_neighbor_sum(
                        i=i_loc_3, j=j, relax_v=relax_v,
                        vx=vx, vy=vy, rhs=vy_rhs,
                        vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4, vy_c5=vy_c5,
                        vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4)

            i_loc_4 = i - 6
            if 1 <= i_loc_4 <= ny1 - 1:
                j_start = 2 if i_loc_4 % 2 == 0 else 1  # Black pass starts on odd (i+j)
                for j in range(j_start, nx1 - 2, 2):
                    vy_c1, vy_c2, vy_c3, vy_c4, vy_c5, vx_c1, vx_c2, vx_c3, vx_c4 = vy_op.compute_coeffs(
                        i=i_loc_4, j=j, dx=dx, dy=dy, etap=etap, etab=etab)

                    vy[i_loc_4, j] = vy_op.compute_neighbor_sum(
                        i=i_loc_4, j=j, relax_v=relax_v,
                        vx=vx, vy=vy, rhs=vy_rhs,
                        vy_c1=vy_c1, vy_c2=vy_c2, vy_c3=vy_c3, vy_c4=vy_c4, vy_c5=vy_c5,
                        vx_c1=vx_c1, vx_c2=vx_c2, vx_c3=vx_c3, vx_c4=vx_c4)

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
