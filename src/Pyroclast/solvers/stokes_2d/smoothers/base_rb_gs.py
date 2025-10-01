"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/model/stokes_2D_mg/smoothers/base_rb_gs.py
Description: File contains copied version of Marcel's Red Black-Gauss-Seidel smoother

Author: Marcel Ferrari, Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""     


import os.path
from typing import Type

import numba as nb
import numpy as np

from Pyroclast.solvers.stokes_2d.smoothers.inline_routines import cpu_inline_loop_body_vx, cpu_inline_loop_body_vy
from Pyroclast.model.stokes_2D_mg.utils import cpu_apply_vx_BC, cpu_apply_vy_BC


use_fast_math_cpu = os.environ.get("PYROCLAST_FASTMATH_CPU", default=False)


@nb.njit(cache=True, parallel=True, fastmath=use_fast_math_cpu)
def _vx_rb_gs_sweep(nx1, ny1,
                    dx, dy,
                    etap, etab,
                    vx, vy,
                    relax_v, rhs, BC) -> np.ndarray:
    """
    In-place Red-Black Gauss-Seidel update for vx.
    """

    # ----------------------------
    #  Red pass: (i + j) % 2 == 0
    # ----------------------------
    for i in nb.prange(1, ny1 - 1):
        j_start = 1 if i % 2 == 0 else 2  # Red pass starts on even (i+j)
        for j in range(j_start, nx1 - 2, 2):
            # Gauss-Seidel in-place update
            vx[i, j] = cpu_inline_loop_body_vx(i=i, j=j,
                                               dx=dx, dy=dy, relax_v=relax_v,
                                               etab=etab, etap=etap,
                                               vx=vx, vy=vy, rhs=rhs)
    # Apply vx boundary conditions
    cpu_apply_vx_BC(vx, BC)

    # ----------------------------
    #  Black pass: (i + j) % 2 == 1
    # ----------------------------
    for i in nb.prange(1, ny1 - 1):
        j_start = 2 if i % 2 == 0 else 1  # Black pass starts on odd (i+j)
        for j in range(j_start, nx1 - 2, 2):
            vx[i, j] = cpu_inline_loop_body_vx(i=i, j=j,
                                               dx=dx, dy=dy, relax_v=relax_v,
                                               etab=etab, etap=etap,
                                               vx=vx, vy=vy, rhs=rhs)

    # Apply vx boundary conditions
    cpu_apply_vx_BC(vx, BC)

    return vx


@nb.njit(cache=True, parallel=True, fastmath=use_fast_math_cpu)
def _vy_red_black_gs_sweep(nx1, ny1,
                           dx, dy,
                           etap, etab,
                           vx, vy,
                           relax_v, rhs, BC) -> np.ndarray:
    """
    In-place Red-Black Gauss-Seidel update for vy.
    """

    # ----------------------------
    #  Red pass
    # ----------------------------
    for i in nb.prange(1, ny1 - 2):
        for j in range(1, nx1 - 1):
            if (i + j) % 2 == 0:

                # 4) Gauss-Seidel in-place update
                vy[i, j] = cpu_inline_loop_body_vy(i=i, j=j,
                                                   dx=dx, dy=dy, relax_v=relax_v,
                                                   etab=etab, etap=etap,
                                                   vx=vx, vy=vy, rhs=rhs)
    # Apply vy boundary conditions
    cpu_apply_vy_BC(vy, BC)

    # ----------------------------
    #  Black pass
    # ----------------------------
    for i in nb.prange(1, ny1 - 2):
        for j in range(1, nx1 - 1):
            if (i + j) % 2 == 1:
                # 4) Gauss-Seidel in-place update
                vy[i, j] = cpu_inline_loop_body_vy(i=i, j=j,
                                                   dx=dx, dy=dy, relax_v=relax_v,
                                                   etab=etab, etap=etap,
                                                   vx=vx, vy=vy, rhs=rhs)

    # Apply vy boundary conditions
    cpu_apply_vy_BC(vy, BC)

    return vy


@nb.njit(cache=True, fastmath=use_fast_math_cpu)
def velocity_smoother_rb_gs(nx1: int, ny1: int,
                            dx: float, dy: float,
                            etap: np.ndarray, etab: np.ndarray,
                            vx: np.ndarray, vy: np.ndarray,
                            relax_v: float, BC: float,
                            vx_rhs: np.ndarray, vy_rhs: np.ndarray, max_iter: int):
    """
    Full Uzawa smoother for velocity and pressure.
    """
    for _ in range(max_iter):
        vx = _vx_rb_gs_sweep(nx1, ny1,
                             dx, dy,
                             etap, etab,
                             vx, vy,
                             relax_v, vx_rhs, BC)

        vy = _vy_red_black_gs_sweep(nx1, ny1,
                                    dx, dy,
                                    etap, etab,
                                    vx, vy,
                                    relax_v, vy_rhs, BC)

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

    class BaseImplementationVX(bw.BenchmarkVX):
        def benchmark_preamble(self):
            start = dtf()
            _vx_rb_gs_sweep(nx1=self.nx1, ny1=self.ny1,
                            dx=self.dx, dy=self.dy,
                            etap=self.eta_p, etab=self.eta_b,
                            vx=self.vx, vy=self.vy,
                            relax_v=self.relax_v, BC=self.boundary_condition, rhs=self.vx_rhs)
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
                _vx_rb_gs_sweep(nx1=self.nx1, ny1=self.ny1,
                                dx=self.dx, dy=self.dy,
                                etap=self.eta_p, etab=self.eta_b,
                                vx=self.vx, vy=self.vy,
                                relax_v=self.relax_v, BC=self.boundary_condition, rhs=self.vx_rhs)
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Benchmark",
                                       stage=Stage.BENCHMARK,
                                       start=start,
                                       end=end))

    class BaseImplementationVY(bw.BenchmarkVY):
        def benchmark_preamble(self):
            start = dtf()
            _vy_red_black_gs_sweep(nx1=self.nx1, ny1=self.ny1,
                                   dx=self.dx, dy=self.dy,
                                   etap=self.eta_p, etab=self.eta_b,
                                   vx=self.vx, vy=self.vy,
                                   relax_v=self.relax_v, BC=self.boundary_condition, rhs=self.vy_rhs)
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
                _vy_red_black_gs_sweep(nx1=self.nx1, ny1=self.ny1,
                                       dx=self.dx, dy=self.dy,
                                       etap=self.eta_p, etab=self.eta_b,
                                       vx=self.vx, vy=self.vy,
                                       relax_v=self.relax_v, BC=self.boundary_condition, rhs=self.vy_rhs)
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Benchmark",
                                       stage=Stage.BENCHMARK,
                                       start=start,
                                       end=end))

    return BaseImplementationBenchmarkSmoother, BaseImplementationVX, BaseImplementationVY
