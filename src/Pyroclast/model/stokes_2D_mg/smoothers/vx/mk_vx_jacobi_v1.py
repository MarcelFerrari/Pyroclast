import math
import os.path
from typing import Type

import numba as nb
import numpy as np

from Pyroclast.model.stokes_2D_mg.utils import apply_vx_BC
from .inline_vx import compute_coeffs, compute_neighbor_sum


"""
Jacobi Implementation of the same algorithm v1 cache locality algorithm
"""
# INFO: Same story as before, I think that this algorith might work good for small problems but not for big problems.


@nb.njit(cache=True, parallel=True)
def vx_jacobi_sweep(nx1: int, ny1: int,
                    dx: float, dy: float,
                    etap: np.ndarray, etab: np.ndarray,
                    vx: np.ndarray, vy: np.ndarray,
                    vx_new: np.ndarray,
                    relax_v:float, BC: float, rhs: np.ndarray,
                    th: int, cache_a: int):
    # Loop only over the interior cells
    for p in nb.prange(th):
        start_y = (ny1 - 2) / th * p + 1
        end_y = ny1 - 1 if p + 1 == th else (ny1 - 2) / th * (p + 1) + 1

        blocks = math.ceil((end_y - start_y) / cache_a)

        for b in range(blocks):
            start_b = start_y + b * cache_a
            end_b = end_y if blocks == b + 1 else start_y + (b + 1) * cache_a

            for j in range(1, nx1 - 2):
                for i in range(start_b, end_b):
                    vx_c1, vx_c2, vx_c3, vx_c4, vx_c5, vy_c1, vy_c2, vy_c3, vy_c4 = compute_coeffs(i, j, dx, dy, etap, etab)

                    # Jacobi update
                    vx_new[i, j] = compute_neighbor_sum(
                        i, j, relax_v, vx_c1, vx_c2, vx_c3, vx_c4, vx_c5, vy_c1, vy_c2, vy_c3, vy_c4, vx, vy, rhs
                    )
    # Copy solution to vx
    vx[:, :] = vx_new[:, :]

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
        needs_cache_block_size_1: bool = True

        def __init__(self, arguments: bw.BenchmarkValidatorVX):
            super().__init__(arguments=arguments)

            if self.vx_new is None:
                self.vx_new = np.zeros((self.nx1, self.ny1))

        def benchmark_preamble(self):
            th = nb.get_num_threads()
            start = dtf()
            vx_jacobi_sweep(nx1=self.nx1, ny1=self.ny1,
                            dx=self.dx, dy=self.dy,
                            etap=self.eta_p, etab=self.eta_b,
                            vx=self.vx, vy=self.vy, vx_new=self.vx_new,
                            relax_v=self.relax_v, BC=self.boundary_condition, rhs=self.vx_rhs,
                            cache_a=self.cache_block_size_1, th=th)
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
                vx_jacobi_sweep(nx1=self.nx1, ny1=self.ny1,
                                dx=self.dx, dy=self.dy,
                                etap=self.eta_p, etab=self.eta_b,
                                vx=self.vx, vy=self.vy, vx_new=self.vx_new,
                                relax_v=self.relax_v, BC=self.boundary_condition, rhs=self.vx_rhs,
                                cache_a=self.args.cache_block_size_1, th=th)
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Benchmark",
                                       stage=Stage.BENCHMARK,
                                       start=start,
                                       end=end))

    return None, BaseImplementationVX, None