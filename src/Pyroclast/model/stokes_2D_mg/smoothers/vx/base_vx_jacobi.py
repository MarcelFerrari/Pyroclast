import os.path
from typing import Type

import numba as nb
import numpy as np

from Pyroclast.model.stokes_2D_mg.utils import apply_vx_BC

# Jacobi update for vx
@nb.njit(cache=True, parallel=True)
def vx_jacobi_sweep(nx1: int, ny1: int,
                    dx: float, dy: float,
                    etap: np.ndarray, etab: np.ndarray,
                    vx: np.ndarray, vy: np.ndarray,
                    vx_new: np.ndarray,
                    relax_v:float, BC: float, rhs: np.ndarray):
    # Loop only over the interior cells
    for i in nb.prange(1, ny1 - 1):
        for j in range(1, nx1 - 2):
            # Gather local (i,j) viscosity coefficients
            etaA = etap[i, j]
            etaB = etap[i, j + 1]
            eta1 = etab[i - 1, j]
            eta2 = etab[i, j]

            # Construct coefficients for x-momentum
            vx1_coeff = 2.0 * etaA / (dx * dx)
            vx2_coeff = eta1 / (dy * dy)
            vx3_coeff = - (eta1 + eta2) / (dy * dy) - 2.0 * (etaA + etaB) / (dx * dx)
            vx4_coeff = eta2 / (dy * dy)
            vx5_coeff = 2.0 * etaB / (dx * dx)

            # Cross terms with vy
            vy1_coeff = eta1 / (dx * dy)
            vy2_coeff = -eta2 / (dx * dy)
            vy3_coeff = -eta1 / (dx * dy)
            vy4_coeff = eta2 / (dx * dy)

            # Sum neighbor contributions for Jacobi
            sum_neighbors = (
                    vx1_coeff * vx[i, j - 1] +
                    vx2_coeff * vx[i - 1, j] +
                    vx4_coeff * vx[i + 1, j] +
                    vx5_coeff * vx[i, j + 1]
                    +
                    vy1_coeff * vy[i - 1, j] +
                    vy2_coeff * vy[i, j] +
                    vy3_coeff * vy[i - 1, j + 1] +
                    vy4_coeff * vy[i, j + 1]
            )

            diag = vx3_coeff

            new_val = (1.0 - relax_v) * vx[i, j] + relax_v * (rhs[i, j] - sum_neighbors) / diag

            # Jacobi update
            vx_new[i, j] = new_val

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
        def __init__(self, arguments: bw.BenchmarkValidatorVX):
            super().__init__(arguments=arguments)
            if self.vx_new is None:
                self.vx_new = np.zeros((self.nx1, self.ny1))

        def benchmark_preamble(self):
            start = dtf()
            vx_jacobi_sweep(nx1=self.nx1, ny1=self.ny1,
                            dx=self.dx, dy=self.dy,
                            etap=self.eta_p, etab=self.eta_b,
                            vx=self.vx, vy=self.vy, vx_new=self.vx_new,
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
                vx_jacobi_sweep(nx1=self.nx1, ny1=self.ny1,
                                dx=self.dx, dy=self.dy,
                                etap=self.eta_p, etab=self.eta_b,
                                vx=self.vx, vy=self.vy, vx_new=self.vx_new,
                                relax_v=self.relax_v, BC=self.boundary_condition, rhs=self.vx_rhs)
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__class__.__name__}: Benchmark",
                                       stage=Stage.BENCHMARK,
                                       start=start,
                                       end=end))

    return None, BaseImplementationVX, None