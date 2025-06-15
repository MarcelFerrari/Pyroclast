import os.path
from typing import Type

import numba as nb
import numpy as np

from Pyroclast.model.stokes_2D_mg.utils import apply_vx_BC, apply_vy_BC
from .inline_vx import compute_coeffs, compute_neighbor_sum


"""
This is a second caching iteration of performing a red-black gauss seidel. 
"""


@nb.njit(cache=True, parallel=True)
def _vx_rb_gs_sweep(nx1, ny1,
                    dx, dy,
                    etap, etab,
                    vx, vy,
                    relax_v, rhs, BC,
                    cache_a: int) -> np.ndarray:
    """
    In-place Red-Black Gauss-Seidel update for vx.
    """

    # ----------------------------
    #  Red pass: (i + j) % 2 == 0
    # ----------------------------
    for i1 in nb.prange(1, ny1 - 1, cache_a):
        for i2 in range(cache_a):
            i = i1 + i2
            j_start = 1 if i % 2 == 0 else 2  # Red pass starts on even (i+j)
            for j in range(j_start, nx1 - 2, 2):
                vx_c1, vx_c2, vx_c3, vx_c4, vx_c5, vy_c1, vy_c2, vy_c3, vy_c4 = compute_coeffs(i, j, dx, dy, etap, etab)

                # Gauss-Seidel in-place update
                vx[i, j] = compute_neighbor_sum(
                    i, j, relax_v, vx_c1, vx_c2, vx_c3, vx_c4, vx_c5, vy_c1, vy_c2, vy_c3, vy_c4, vx, vy, rhs
                )

    # Apply vx boundary conditions
    apply_vx_BC(vx, BC)

    # ----------------------------
    #  Black pass: (i + j) % 2 == 1
    # ----------------------------
    for i1 in nb.prange(1, ny1 - 1, cache_a):
        for i2 in range(cache_a):
            i = i1 + i2
            j_start = 2 if i % 2 == 0 else 1  # Black pass starts on odd (i+j)
            for j in range(j_start, nx1 - 2, 2):
                vx_c1, vx_c2, vx_c3, vx_c4, vx_c5, vy_c1, vy_c2, vy_c3, vy_c4 = compute_coeffs(i, j, dx, dy, etap, etab)

                # Gauss-Seidel in-place update
                vx[i, j] = compute_neighbor_sum(
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
    from benchmark.benchmark_validators import Stage, Timing
    from benchmark.utils import dtf

    module_name = os.path.basename(__file__).replace(".py", "")

    class BaseImplementationVX(bw.BenchmarkVX):
        def benchmark_preamble(self):
            start = dtf()
            _vx_rb_gs_sweep(nx1=self.nx1, ny1=self.ny1,
                            dx=self.dx, dy=self.dy,
                            etap=self.eta_p, etab=self.eta_b,
                            vx=self.vx, vy=self.vy,
                            relax_v=self.relax_v, BC=self.boundary_condition, rhs=self.vx_rhs,
                            cache_a=self.args.cache_block_size_1)
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
                                relax_v=self.relax_v, BC=self.boundary_condition, rhs=self.vx_rhs,
                                cache_a=self.args.cache_block_size_1)
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Benchmark",
                                       stage=Stage.BENCHMARK,
                                       start=start,
                                       end=end))

    return None, BaseImplementationVX, None