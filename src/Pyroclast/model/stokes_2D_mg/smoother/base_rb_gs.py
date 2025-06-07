import os.path
from typing import Type

import numba as nb
import numpy as np

from Pyroclast.model.stokes_2D_mg.utils import apply_vx_BC, apply_vy_BC


@nb.njit(cache=True, parallel=True)
def _vx_rb_gs_sweep(nx1, ny1,
                    dx, dy,
                    etap, etab,
                    vx, vy,
                    relax_v, rhs, BC):
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

            # TODO lift
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

            # 3) Sum neighbor contributions
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

            # Gauss-Seidel in-place update
            vx[i, j] = (1.0 - relax_v) * vx[i, j] \
                       + relax_v * (rhs[i, j] - sum_neighbors) / diag
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

            # TODO lift
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

            # 3) Sum neighbor contributions

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

            # Gauss-Seidel in-place update
            vx[i, j] = (1.0 - relax_v) * vx[i, j] \
                       + relax_v * (rhs[i, j] - sum_neighbors) / diag

    # Apply vx boundary conditions
    apply_vx_BC(vx, BC)

    return vx


@nb.njit(cache=True, parallel=True)
def _vy_red_black_gs_sweep(nx1, ny1,
                           dx, dy,
                           etap, etab,
                           vx, vy,
                           relax_v, rhs, BC):
    """
    In-place Red-Black Gauss-Seidel update for vy.
    """

    # ----------------------------
    #  Red pass
    # ----------------------------
    for i in nb.prange(1, ny1 - 2):
        for j in range(1, nx1 - 1):
            if (i + j) % 2 == 0:
                # 1) Gather local viscosities
                etaA = etap[i, j]
                etaB = etap[i + 1, j]
                eta1 = etab[i, j - 1]
                eta2 = etab[i, j]

                # TODO lift
                # 2) Construct coefficients for y-momentum
                vy1_coeff = eta1 / (dx * dx)
                vy2_coeff = 2.0 * etaA / (dy * dy)
                vy3_coeff = -2.0 * etaA / (dy * dy) \
                            - 2.0 * etaB / (dy * dy) \
                            - eta1 / (dx * dx) \
                            - eta2 / (dx * dx)
                vy4_coeff = 2.0 * etaB / (dy * dy)
                vy5_coeff = eta2 / (dx * dx)

                # Cross terms with vx
                vx1_coeff = eta1 / (dx * dy)
                vx2_coeff = -eta1 / (dx * dy)
                vx3_coeff = -eta2 / (dx * dy)
                vx4_coeff = eta2 / (dx * dy)

                # 3) Sum neighbor contributions
                sum_neighbors = (
                    # vy neighbors
                        vy1_coeff * vy[i, j - 1] +
                        vy2_coeff * vy[i - 1, j] +
                        vy4_coeff * vy[i + 1, j] +
                        vy5_coeff * vy[i, j + 1]
                        +
                        # cross terms with vx
                        vx1_coeff * vx[i, j - 1] +
                        vx2_coeff * vx[i + 1, j - 1] +
                        vx3_coeff * vx[i, j] +
                        vx4_coeff * vx[i + 1, j]
                )

                diag = vy3_coeff

                # 4) Gauss-Seidel in-place update
                vy[i, j] = (1.0 - relax_v) * vy[i, j] \
                           + relax_v * (rhs[i, j] - sum_neighbors) / diag
    # Apply vy boundary conditions
    apply_vy_BC(vy, BC)

    # ----------------------------
    #  Black pass
    # ----------------------------
    for i in nb.prange(1, ny1 - 2):
        for j in range(1, nx1 - 1):
            if (i + j) % 2 == 1:
                # 1) Gather viscosities
                etaA = etap[i, j]
                etaB = etap[i + 1, j]
                eta1 = etab[i, j - 1]
                eta2 = etab[i, j]

                # TODO lift
                # 2) Coefficients for y-momentum
                vy1_coeff = eta1 / (dx * dx)
                vy2_coeff = 2.0 * etaA / (dy * dy)
                vy3_coeff = -2.0 * etaA / (dy * dy) \
                            - 2.0 * etaB / (dy * dy) \
                            - eta1 / (dx * dx) \
                            - eta2 / (dx * dx)
                vy4_coeff = 2.0 * etaB / (dy * dy)
                vy5_coeff = eta2 / (dx * dx)

                # Cross terms with vx
                vx1_coeff = eta1 / (dx * dy)
                vx2_coeff = -eta1 / (dx * dy)
                vx3_coeff = -eta2 / (dx * dy)
                vx4_coeff = eta2 / (dx * dy)

                # 3) Sum neighbors
                sum_neighbors = (
                        vy1_coeff * vy[i, j - 1] +
                        vy2_coeff * vy[i - 1, j] +
                        vy4_coeff * vy[i + 1, j] +
                        vy5_coeff * vy[i, j + 1]
                        +
                        vx1_coeff * vx[i, j - 1] +
                        vx2_coeff * vx[i + 1, j - 1] +
                        vx3_coeff * vx[i, j] +
                        vx4_coeff * vx[i + 1, j]
                )

                diag = vy3_coeff

                # 4) In-place update
                vy[i, j] = (1.0 - relax_v) * vy[i, j] \
                           + relax_v * (rhs[i, j] - sum_neighbors) / diag

    # Apply vy boundary conditions
    apply_vy_BC(vy, BC)

    return vy


@nb.njit(cache=True)
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
            self.timings.append(Timing(name=f"{module_name}.{self.__name__}: Preamble",
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
            for _ in range(self.args.samples):
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
                self.timings.append(Timing(name=f"{module_name}.{self.__name__}: Benchmark",
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
                            relax_v=self.relax_v, BC=self.boundary_condition, rhs=self.vx_rhs,
                            max_iter=1)
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__name__}: Preamble",
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
            for _ in range(self.args.samples):
                start = dtf()
                _vx_rb_gs_sweep(nx1=self.nx1, ny1=self.ny1,
                                dx=self.dx, dy=self.dy,
                                etap=self.eta_p, etab=self.eta_b,
                                vx=self.vx, vy=self.vy,
                                relax_v=self.relax_v, BC=self.boundary_condition, rhs=self.vx_rhs,
                                max_iter=1)
                end = dtf()

                # Add the timing information
                self.timings.append(Timing(name=f"{module_name}.{self.__name__}: Benchmark",
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
                                   relax_v=self.relax_v, BC=self.boundary_condition, rhs=self.vy_rhs,
                                   max_iter=1)
            end = dtf()

            # Add the timing information
            self.timings.append(Timing(name=f"{module_name}.{self.__name__}: Preamble",
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
            for _ in range(self.args.samples):
                start = dtf()
                _vy_red_black_gs_sweep(nx1=self.nx1, ny1=self.ny1,
                                       dx=self.dx, dy=self.dy,
                                       etap=self.eta_p, etab=self.eta_b,
                                       vx=self.vx, vy=self.vy,
                                       relax_v=self.relax_v, BC=self.boundary_condition, rhs=self.vy_rhs,
                                       max_iter=1)
                end = dtf()

                # Add the timing information
                self.timings.append(Timing(name=f"{module_name}.{self.__name__}: Benchmark",
                                           stage=Stage.BENCHMARK,
                                           start=start,
                                           end=end))

    return BaseImplementationBenchmarkSmoother, BaseImplementationVX, BaseImplementationVY