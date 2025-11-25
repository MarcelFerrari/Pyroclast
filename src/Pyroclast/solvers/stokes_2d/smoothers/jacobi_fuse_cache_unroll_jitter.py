import os
from typing import Type

import numba as nb
import numpy as np

from Pyroclast.utils import inject_threads
from .inline_routines import cpu_inline_loop_body_vx, cpu_inline_loop_body_vy


@nb.njit(inline='always')
def apply_vx_BC_ij(vx: np.ndarray, i: int, j: int, BC: float):
    """
    Apply ONLY the boundary entries touched by interior (i,j) for vx.
    vx interior: i in [1..ny1-2], j in [1..nx1-3]
    """
    ny1, nx1 = vx.shape

    # Top / bottom depend on neighbor interior rows
    if i == 1:  # touching top boundary row
        vx[0, j] = -BC * vx[1, j]
    if i == ny1 - 2:  # touching bottom boundary row
        vx[ny1 - 1, j] = -BC * vx[ny1 - 2, j]

    # Left wall
    if j == 1:  # touching left boundary column
        vx[i, 0] = 0.0

    # Right + ghost columns
    if j == nx1 - 3:  # touching right boundary / ghost
        vx[i, nx1 - 2] = 0.0
        vx[i, nx1 - 1] = 0.0


@nb.njit(inline='always')
def apply_vy_BC_ij(vy: np.ndarray, i: int, j: int, BC: float):
    """
    Apply ONLY the boundary entries touched by interior (i,j) for vy.
    vy interior: i in [1..ny1-3], j in [1..nx1-2]
    """
    ny1, nx1 = vy.shape

    # Left / right depend on neighbor interior columns
    if j == 1:  # touching left boundary column
        vy[i, 0] = -BC * vy[i, 1]
    if j == nx1 - 2:  # touching right boundary column
        vy[i, nx1 - 1] = -BC * vy[i, nx1 - 2]

    # Top wall
    if i == 1:  # touching top boundary row
        vy[0, j] = 0.0

    # Bottom + ghost rows
    if i == ny1 - 3:  # touching bottom boundary / ghost
        vy[ny1 - 2, j] = 0.0
        vy[ny1 - 1, j] = 0.0


@nb.njit(cache=True)
def _vx_tile_kernel(ii: int, jj: int,
                    nx1: int, ny1: int,
                    dx: float, dy: float,
                    etap: np.ndarray, etab: np.ndarray,
                    vy: np.ndarray, vx_rhs: np.ndarray,
                    vx_src: np.ndarray, vx_dst: np.ndarray,
                    relax_v: float, BC: float, iter_unroll: int ,cache_stride: int, cache_time: int):
    """
    Process one vx tile [ii:ii+cache_time) x [jj:jj+cache_stride) with iter_unroll Jacobi steps.
    Reads vy (constant for this tile pass), ping-pongs between (vx_src, vx_dst) locally.
    """
    # vx interior: i in [1..ny1-2], j in [1..nx1-3]
    i_max = min(ii + cache_time, ny1 - 1)  # stop before ny1-1 -> last i = ny1-2
    j_max = min(jj + cache_stride, nx1 - 2)  # stop before nx1-2 -> last j = nx1-3

    # Local references we can swap without affecting caller
    src = vx_src
    dst = vx_dst

    for _ in range(iter_unroll):
        for i in range(ii, i_max):
            # ensure we don't run into the guard rows
            for j in range(jj, j_max):
                dst[i, j] = cpu_inline_loop_body_vx(i=i, j=j, relax_v=relax_v,
                                                    dx=dx, dy=dy,
                                                    etab=etab, etap=etap,
                                                    vx=src, vy=vy, rhs=vx_rhs)
                apply_vx_BC_ij(dst, i, j, BC)  # apply BCs for this tile
        src, dst = dst, src

    # No return; writes already in src/dst. For even iter_unroll, results end in original vx_src.


@nb.njit(cache=True)
def _vy_tile_kernel(ii: int, jj: int,
                    nx1: int, ny1: int,
                    dx: float, dy: float,
                    etap: np.ndarray, etab: np.ndarray,
                    vx: np.ndarray, vy_rhs: np.ndarray,
                    vy_src: np.ndarray, vy_dst: np.ndarray,
                    relax_v: float, BC: float, iter_unroll: int, cache_stride: int, cache_time: int):
    """
    Process one vy tile [ii:ii+cache_time) x [jj:jj+cache_stride) with iter_unroll Jacobi steps.
    Reads vx (constant for this tile pass), ping-pongs between (vy_src, vy_dst) locally.
    """
    # vy interior: i in [1..ny1-3], j in [1..nx1-2]
    i_max = min(ii + cache_time, ny1 - 2)  # stop before ny1-2 -> last i = ny1-3
    j_max = min(jj + cache_stride, nx1 - 1)  # stop before nx1-1 -> last j = nx1-2

    src = vy_src
    dst = vy_dst

    for _ in range(iter_unroll):
        for i in range(ii, i_max):
            # inner loop is a good vectorization candidate for LLVM
            for j in range(jj, j_max):
                dst[i, j] = cpu_inline_loop_body_vy(i=i, j=j, relax_v=relax_v,
                                                    dx=dx, dy=dy,
                                                    etab=etab, etap=etap,
                                                    vy=src, vx=vx, rhs=vy_rhs)
                apply_vy_BC_ij(dst, i, j, BC)  # apply BCs for this tile

        src, dst = dst, src  # local ping-pong


@inject_threads
@nb.njit(cache=True, parallel=True)
def ras_velocity_jacobi_smoother(nx1: int, ny1: int,
                                 dx: float, dy: float,
                                 etap: np.ndarray, etab: np.ndarray,
                                 vx: np.ndarray, vy: np.ndarray,
                                 relax_v: float, BC: float,
                                 vx_rhs: np.ndarray, vy_rhs: np.ndarray,
                                 max_iter: int, cache_stride: int, cache_time: int, iter_unroll: int,
                                 vx_old:np.ndarray, vy_old:np.ndarray,
                                 th: int):
    max_iter += max_iter % 2
    outer_iters = (max_iter + iter_unroll) // (iter_unroll + 1)

    # coverage under half-tile shifts
    # randint(0, TILE//2 + 1) -> smax = TILE//2
    smax_i = cache_time
    smax_j = cache_stride
    Ni = (ny1 - 2)
    Nj = (nx1 - 2)
    tiles_i = (Ni + smax_i + cache_time - 1) // cache_time
    tiles_j = (Nj + smax_j + cache_stride - 1) // cache_stride
    total_tiles = tiles_i * tiles_j

    workers = th if total_tiles >= th else total_tiles

    for out in range(outer_iters):
        # phase shift
        shift_i = np.random.randint(0, smax_i + 1)
        shift_j = np.random.randint(0, smax_j + 1)
        # shift_i = 0
        # shift_j = 0
        ii0 = 1 - shift_i
        jj0 = 1 - shift_j

        # contiguous, balanced blocks (no rotation)
        base = total_tiles // workers
        rem = total_tiles - base * workers

        for lane in nb.prange(workers):
            # first 'rem' lanes get +1 tile
            gain = 1 if lane < rem else 0
            start = lane * base + (lane if lane < rem else rem)
            count = base + gain

            # turn linear start into (ti,tj) once, then walk contiguously
            k = start
            ti = k // tiles_j
            tj = k - ti * tiles_j

            for _ in range(count):
                ii = ii0 + ti * cache_time
                jj = jj0 + tj * cache_stride
                if ii < 1: ii = 1
                if jj < 1: jj = 1

                _vx_tile_kernel(ii=ii, jj=jj, nx1=nx1, ny1=ny1, dx=dx, dy=dy,
                                etap=etap, etab=etab, vy=vy, vx_rhs=vx_rhs,
                                vx_src=vx, vx_dst=vx_old, relax_v=relax_v, BC=BC,
                                cache_stride=cache_stride, cache_time=cache_time, iter_unroll=iter_unroll)

                _vy_tile_kernel(ii=ii, jj=jj, nx1=nx1, ny1=ny1, dx=dx, dy=dy,
                                etap=etap, etab=etab, vx=vx_old, vy_rhs=vy_rhs,
                                vy_src=vy, vy_dst=vy_old, relax_v=relax_v, BC=BC,
                                cache_stride=cache_stride, cache_time=cache_time, iter_unroll=iter_unroll)

                # advance to next tile, wrap across rows
                tj += 1
                if tj == tiles_j:
                    tj = 0
                    ti += 1

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
        needs_cache_block_size_1: bool = True

        def __init__(self, arguments: bw.BenchmarkValidatorSmoother):
            super().__init__(arguments=arguments)

            if self.vx_new is None:
                self.vx_new = np.zeros((self.nx1, self.ny1))

            if self.vy_new is None:
                self.vy_new = np.zeros((self.nx1, self.ny1))

        def benchmark_preamble(self):
            start = dtf()
            ras_velocity_jacobi_smoother(nx1=self.nx1, ny1=self.ny1,
                                         dx=self.dx, dy=self.dy,
                                         etap=self.eta_p, etab=self.eta_b,
                                         vx=self.vx, vy=self.vy, vx_new=self.vx_new, vy_new=self.vy_new,
                                         relax_v=self.relax_v, BC=self.boundary_condition,
                                         max_iter=1,
                                         vx_rhs=self.vx_rhs, vy_rhs=self.vy_rhs,
                                         cache_a=self.cache_block_size_1)
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
            ras_velocity_jacobi_smoother(nx1=self.nx1, ny1=self.ny1,
                                         dx=self.dx, dy=self.dy,
                                         etap=self.eta_p, etab=self.eta_b,
                                         vx=self.vx, vy=self.vy, vx_new=self.vx_new, vy_new=self.vy_new,
                                         relax_v=self.relax_v, BC=self.boundary_condition,
                                         max_iter=self.max_iter,
                                         vx_rhs=self.vx_rhs, vy_rhs=self.vy_rhs,
                                         cache_a=self.cache_block_size_1)
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Benchmark",
                                       stage=Stage.BENCHMARK,
                                       start=start,
                                       end=end))

    # INFO: Methods can be benchmarked in other places. Here is only the full implementation.
    return BaseImplementationBenchmarkSmoother, None, None
