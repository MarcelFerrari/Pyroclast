import os.path
from typing import Type

import numba as nb
import numpy as np

from Pyroclast.model.stokes_2D_mg.utils import apply_vx_BC, apply_vy_BC

"""
This file aims to test the decorator functionality of always inlining a given function. This serves as a quick sanity
check that inlining doesn't degrade performance
"""

@nb.njit(cache=True, inline="always")
def compute_coeffs(i: int, j: int,
                   dx: float, dy: float,
                   etap: np.ndarray, etab: np.ndarray):
    """
    External compute coeffs function to reduce on code duplication
    """
    # 1) Gather local viscosities
    etaA = etap[i, j]
    etaB = etap[i, j + 1]
    eta1 = etab[i - 1, j]
    eta2 = etab[i, j]

    # 2) Construct coefficients for x-momentum
    vx1_coeff = 2.0 * etaA / (dx * dx)
    vx2_coeff = eta1 / (dy * dy)
    vx3_coeff = -(eta1 + eta2) / (dy * dy) \
                - 2.0 * (etaA + etaB) / (dx * dx)
    vx4_coeff = eta2 / (dy * dy)
    vx5_coeff = 2.0 * etaB / (dx * dx)

    # Cross terms with vy
    vy1_coeff = eta1 / (dx * dy)
    vy2_coeff = -eta2 / (dx * dy)
    vy3_coeff = -eta1 / (dx * dy)
    vy4_coeff = eta2 / (dx * dy)

    return vx1_coeff, vx2_coeff, vx3_coeff, vx4_coeff, vx5_coeff, vy1_coeff, vy2_coeff, vy3_coeff, vy4_coeff


@nb.njit(cache=True, inline="always")
def compute_neighbor_sum(i: int, j: int, relax_v: float,
                         vx_c1: float, vx_c2: float, vx_c3: float, vx_c4: float, vx_c5: float,
                         vy_c1: float, vy_c2: float, vy_c3: float, vy_c4: float,
                         vx: np.ndarray, vy: np.ndarray, rhs: np.ndarray) -> float:

    # 3) Sum neighbor contributions
    sum_neighbors = (
            vx_c1 * vx[i, j - 1] +
            vx_c2 * vx[i - 1, j] +
            vx_c4 * vx[i + 1, j] +
            vx_c5 * vx[i, j + 1]
            +
            vy_c1 * vy[i - 1, j] +
            vy_c2 * vy[i, j] +
            vy_c3 * vy[i - 1, j + 1] +
            vy_c4 * vy[i, j + 1]
    )

    diag = vx_c3

    return (1.0 - relax_v) * vx[i, j] + relax_v * (rhs[i, j] - sum_neighbors) / diag

@nb.njit(cache=True, parallel=True)
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
    for i in nb.prange(1, ny1 - 1):
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

    return None, BaseImplementationVX, None