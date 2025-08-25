import os.path
from typing import Type

import numba as nb
import numpy as np

from Pyroclast.model.stokes_2D_mg.utils import apply_vx_BC, apply_vy_BC


"""
This file contains an experiment on the optimization capabilities of Numba. The aim was to see if vectorization would 
improve, if the code was restructured in a way that allows for quicker reuse of values.
"""


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
            # 1) Gather local viscosities
            etaA = etap[i, j]
            etaB = etap[i, j + 1]
            eta1 = etab[i - 1, j]
            eta2 = etab[i, j]

            # 2) Construct coefficients for x-momentum
            vx1_coeff = 2.0 * etaA / (dx * dx)
            sn1 = vx1_coeff * vx[i, j - 1]

            vx2_coeff = eta1 / (dy * dy)
            sn2 = sn1 + vx2_coeff * vx[i - 1, j]

            vx4_coeff = eta2 / (dy * dy)
            sn4 = sn2 + vx4_coeff * vx[i + 1, j]

            vx5_coeff = 2.0 * etaB / (dx * dx)
            sn5 = sn4 + vx5_coeff * vx[i, j + 1]

            # Cross terms with vy
            vy1_coeff = eta1 / (dx * dy)
            sn51 = sn5 + vy1_coeff * vy[i - 1, j]

            vy2_coeff = -eta2 / (dx * dy)
            sn52 = sn51 + vy2_coeff * vy[i, j]

            vy3_coeff = -eta1 / (dx * dy)
            sn53 = sn52 + vy3_coeff * vy[i - 1, j + 1]
            vy4_coeff = eta2 / (dx * dy)
            sn54 = sn53 + vy4_coeff * vy[i, j + 1]

            # 3) Sum neighbor contributions
            vx3_coeff = -(eta1 + eta2) / (dy * dy) \
                        - 2.0 * (etaA + etaB) / (dx * dx)

            # Gauss-Seidel in-place update
            vx[i, j] = (1.0 - relax_v) * vx[i, j] \
                       + relax_v * (rhs[i, j] - sn54) / vx3_coeff

    # Apply vx boundary conditions
    apply_vx_BC(vx, BC)

    # ----------------------------
    #  Black pass: (i + j) % 2 == 1
    # ----------------------------
    for i in nb.prange(1, ny1 - 1):
        j_start = 2 if i % 2 == 0 else 1  # Black pass starts on odd (i+j)
        for j in range(j_start, nx1 - 2, 2):
            # 1) Gather local viscosities
            etaA = etap[i, j]
            etaB = etap[i, j + 1]
            eta1 = etab[i - 1, j]
            eta2 = etab[i, j]

            # 2) Construct coefficients for x-momentum
            vx1_coeff = 2.0 * etaA / (dx * dx)
            sn1 = vx1_coeff * vx[i, j - 1]

            vx2_coeff = eta1 / (dy * dy)
            sn2 = sn1 + vx2_coeff * vx[i - 1, j]

            vx4_coeff = eta2 / (dy * dy)
            sn4 = sn2 + vx4_coeff * vx[i + 1, j]

            vx5_coeff = 2.0 * etaB / (dx * dx)
            sn5 = sn4 + vx5_coeff * vx[i, j + 1]

            # Cross terms with vy
            vy1_coeff = eta1 / (dx * dy)
            sn51 = sn5 + vy1_coeff * vy[i - 1, j]

            vy2_coeff = -eta2 / (dx * dy)
            sn52 = sn51 + vy2_coeff * vy[i, j]

            vy3_coeff = -eta1 / (dx * dy)
            sn53 = sn52 + vy3_coeff * vy[i - 1, j + 1]
            vy4_coeff = eta2 / (dx * dy)
            sn54 = sn53 + vy4_coeff * vy[i, j + 1]

            # 3) Sum neighbor contributions
            vx3_coeff = -(eta1 + eta2) / (dy * dy) \
                        - 2.0 * (etaA + etaB) / (dx * dx)

            # Gauss-Seidel in-place update
            vx[i, j] = (1.0 - relax_v) * vx[i, j] \
                       + relax_v * (rhs[i, j] - sn54) / vx3_coeff
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