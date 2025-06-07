from abc import abstractmethod, ABC
from typing import Optional

import numpy as np
import pyinstrument

from .benchmark_validators import (BaseBenchmarkValidator, BenchmarkValidatorVX, BenchmarkValidatorVY,
                                   BenchmarkValidatorSmoother, Timing)

"""
File contains a basic wrapper class that is used to instantiate and run a benchmark.
It takes care of validating the parameters.
"""


class BaseBenchmark:
    nx1: int
    ny1: int

    dx: float
    dy: float

    eta_b: np.ndarray
    eta_p: np.ndarray

    relax_v: float
    boundary_condition: float

    max_iter: int

    timings: list[Timing]

    args: BaseBenchmarkValidator

    def __init__(self,
                 arguments: BaseBenchmarkValidator):
        """
        Class contains base information about a benchmark
        """
        self.max_iter = arguments.max_iter

        self.nx1 = arguments.nx + 1
        self.ny1 = arguments.ny + 1

        self.dx = 1.0 / (self.nx1 -2) if arguments.dx is None else arguments.dx
        self.dy = 1.0 / (self.ny1 -2) if arguments.dy is None else arguments.dy

        self.eta_b = np.random.rand(self.ny1, self.nx1) * 1e19 + 1e19
        self.eta_p = np.random.rand(self.ny1, self.nx1) * 1e19 + 1e19

        self.relax_v = arguments.relax_v
        self.boundary_condition = arguments.boundary_condition

        self.timings = []
        self.args = arguments

    def benchmark(self):
        """
        Actually run the benchmark.
        """
        self.benchmark_preamble()

        # Call benchmark function either with profile wrapper or without depending on arguments
        if self.args.profile:
            with pyinstrument.profile():
                for _ in range(self.args.samples):
                    self.run_benchmark()
        else:
            for _ in range(self.args.samples):
                self.run_benchmark()

        self.benchmark_epilogue()

    @abstractmethod
    def benchmark_preamble(self):
        """
        Do some preparations before running the benchmark
        """
        ...

    @abstractmethod
    def run_benchmark(self):
        """
        Actually run the benchmark.
        """
        ...

    @abstractmethod
    def benchmark_epilogue(self):
        """
        Perform operations after the benchmark is done
        """
        ...


class BenchmarkVX(ABC, BaseBenchmark):
    vx: np.ndarray
    vx_new: Optional[np.ndarray] = None
    vy: np.ndarray
    vx_rhs: np.ndarray

    args: BenchmarkValidatorVX

    def __init__(self, arguments: BenchmarkValidatorVX):
        """
        If the added arguments for the VX case aren't added,
        """
        super().__init__(arguments=arguments)

        self.vx = arguments.vx if arguments.vx is None else np.zeros((self.ny1, self.nx1))
        self.vy = arguments.vy if arguments.vy is None else np.zeros((self.ny1, self.nx1))
        self.vx_rhs  = arguments.vx_rhs if arguments.vx_rhs is None else np.zeros((self.ny1, self.nx1))

        self.vx_new = arguments.vx_new if arguments.vx_new is not None else None

        self.args = arguments


class BenchmarkVY(ABC, BaseBenchmark):
    vy: np.ndarray
    vy_new: Optional[np.ndarray] = None
    vx: np.ndarray
    vy_rhs: np.ndarray

    args: BenchmarkValidatorVY

    def __init__(self, arguments: BenchmarkValidatorVY):
        """
        If the added arguments for the VX case aren't added,
        """
        super().__init__(arguments=arguments)

        self.vy = arguments.vy if arguments.vy is None else np.zeros((self.ny1, self.nx1))
        self.vx = arguments.vx if arguments.vx is None else np.zeros((self.ny1, self.nx1))
        self.vy_rhs  = arguments.vy_rhs if arguments.vy_rhs is None else np.zeros((self.ny1, self.nx1))

        self.vy_new = arguments.vy_new if arguments.vy_new else None

        self.args = arguments


class BenchmarkSmoother(ABC, BaseBenchmark):
    vx: np.ndarray
    vx_new: Optional[np.ndarray] = None
    vx_rhs: np.ndarray

    vy: np.ndarray
    vy_new: Optional[np.ndarray] = None
    vy_rhs: np.ndarray

    args: BenchmarkValidatorSmoother

    def __init__(self, arguments: BenchmarkValidatorSmoother):
        super().__init__(arguments=arguments)

        self.vx = arguments.vx if arguments.vx is None else np.zeros((self.ny1, self.nx1))
        self.vx_rhs  = arguments.vx_rhs if arguments.vx_rhs is None else np.zeros((self.ny1, self.nx1))

        self.vx_new = arguments.vx_new if arguments.vx_new is not None else None

        self.vy = arguments.vy if arguments.vy is None else np.zeros((self.ny1, self.nx1))
        self.vy_rhs  = arguments.vy_rhs if arguments.vy_rhs is None else np.zeros((self.ny1, self.nx1))

        self.vy_new = arguments.vy_new if arguments.vy_new else None

        self.args = arguments