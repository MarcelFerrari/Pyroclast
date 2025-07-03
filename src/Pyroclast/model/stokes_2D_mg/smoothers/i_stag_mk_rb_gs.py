import numba as nb
import numpy as np
import  os
from typing import Type

from Pyroclast.model.stokes_2D_mg.smoothers.vx.mk_sg_vx_rb_gs_v1 import _vx_rb_gs_sweep
from Pyroclast.model.stokes_2D_mg.smoothers.vy.mk_sg_vy_rb_gs_v1 import _vy_rb_gs_sweep
from benchmark.benchmark_validators import BaseBenchmarkValidator

"""
Benchmark the full smoother with the both vx and vy sweep being internally staggered (no staggering in the smoother for 
loop.) Uses v1 staggering algorithm (internal if)
"""


@nb.njit(cache=True)
def velocity_smoother_rb_gs(nx1: int, ny1: int,
                            dx: float, dy: float,
                            etap: np.ndarray, etab: np.ndarray,
                            vx: np.ndarray, vy: np.ndarray,
                            relax_v: float, BC: float,
                            vx_rhs: np.ndarray, vy_rhs: np.ndarray, max_iter: int,
                            th: int, cache: int):
    """
    Full Uzawa smoother for velocity and pressure.
    """
    for _ in range(max_iter):
        vx = _vx_rb_gs_sweep(nx1, ny1,
                             dx, dy,
                             etap, etab,
                             vx, vy,
                             relax_v, vx_rhs, BC,
                             th, cache)

        vy = _vy_rb_gs_sweep(nx1, ny1,
                             dx, dy,
                             etap, etab,
                             vx, vy,
                             relax_v, vy_rhs, BC,
                             th, cache)

    return vx, vy


# INFO need to use string references to avoid circular imports and deal with benchmark packaged not available
def benchmark_factory() -> tuple[Type["BenchmarkSmoother"], Type["BenchmarkVX"], Type["BenchmarkVY"]]:
    """
    Returns Benchmark Classes needed for benchmarking. Done via factory to avoid issues with the `benchmark` package
    not being available in a production environment.
    """
    import benchmark.benchmark_wrapper as bw
    from benchmark.benchmark_validators import Stage, Timing, BenchmarkValidatorSmoother
    from benchmark.utils import dtf

    module_name = os.path.basename(__file__).replace(".py", "")

    class BaseImplementationBenchmarkSmoother(bw.BenchmarkSmoother):
        needs_cache_block_size_1: bool = True

        def __init__(self, arguments: BenchmarkValidatorSmoother):
            super().__init__(arguments=arguments)
            self.cache_block_size_1 = arguments.cache_block_size_1

        def benchmark_preamble(self):
            th = nb.get_num_threads()
            start = dtf()
            velocity_smoother_rb_gs(nx1=self.nx1, ny1=self.ny1,
                                    dx=self.dx, dy=self.dy,
                                    etap=self.eta_p, etab=self.eta_b,
                                    vx=self.vx, vy=self.vy,
                                    relax_v=self.relax_v, BC=self.boundary_condition,
                                    max_iter=1,
                                    vx_rhs=self.vx_rhs, vy_rhs=self.vy_rhs,
                                    th=th, cache=self.cache_block_size_1)
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
            velocity_smoother_rb_gs(nx1=self.nx1, ny1=self.ny1,
                                    dx=self.dx, dy=self.dy,
                                    etap=self.eta_p, etab=self.eta_b,
                                    vx=self.vx, vy=self.vy,
                                    relax_v=self.relax_v, BC=self.boundary_condition,
                                    max_iter=self.max_iter,
                                    vx_rhs=self.vx_rhs, vy_rhs=self.vy_rhs,
                                    th=th, cache=self.cache_block_size_1)
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Benchmark",
                                       stage=Stage.BENCHMARK,
                                       start=start,
                                       end=end))

    # INFO: Methods can be benchmarked in other places. Here is only the full implementation.
    return BaseImplementationBenchmarkSmoother, None, None