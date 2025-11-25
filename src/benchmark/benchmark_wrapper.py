"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: benchmark/benchmark_wrapper.py
Description: File contains a basic wrapper class that is used to instantiate and run a benchmark.
             It takes care of validating the parameters.


Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""     

from abc import abstractmethod, ABC
from typing import Optional

import numba as nb
import numpy as np
import pyinstrument

from .benchmark_validators import (BaseBenchmarkValidator, BenchmarkValidatorVX, BenchmarkValidatorVY,
                                   BenchmarkValidatorSmoother, Timing)


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

    cache_block_size_1: Optional[int]
    cache_block_size_2: Optional[int]
    iter_unroll: Optional[int]
    jitter: Optional[int]

    timings: list[Timing]

    args: BaseBenchmarkValidator

    needs_cache_block_size_1: bool = False
    needs_cache_block_size_2: bool = False
    needs_iter_unroll: bool = False
    needs_jitter: bool = False

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

        self.cache_block_size_1 = arguments.cache_block_size_1
        self.cache_block_size_2 = arguments.cache_block_size_2
        self.iter_unroll = arguments.iter_unroll

        self.validate_self()

    def validate_self(self):
        """
        Validate that supplementary values are provided
        """
        if self.needs_cache_block_size_1 and self.cache_block_size_1 is None:
            raise ValueError("cache_block_size_1 or cache_a is needed.")

        if self.needs_cache_block_size_2 and self.cache_block_size_2 is None:
            raise ValueError("cache_block_size_1 or cache_a is needed.")

        if self.needs_iter_unroll and self.iter_unroll is None:
            raise ValueError("iter_unroll is needed.")

        if self.needs_jitter and self.jitter is None:
            raise ValueError("jitter is needed.")

    def benchmark(self):
        """
        Actually run the benchmark.
        """
        self.benchmark_preamble()

        # Call benchmark function either with profile wrapper or without depending on arguments
        if self.args.profile:
            with pyinstrument.profile():
                for s in range(self.args.samples):
                    print(f"Running Sample {s + 1} of {self.args.samples}")
                    self.run_benchmark()
        else:
            for s in range(self.args.samples):
                print(f"Running Sample {s + 1} of {self.args.samples}")
                self.run_benchmark()

        self.benchmark_epilogue()

    def validate_run_args(self):
        """
        Call this function prior to executing the benchmark function. Checks that the cache block sizes work.
        **Assumes cache_block_size_1 is for y and cache_block_size_2 is for x with iter_unroll**

        """
        th = nb.get_num_threads()
        if self.needs_cache_block_size_1:
            if (self.ny1 / th) < self.cache_block_size_1:
                raise ValueError("cache_block_size_1 must be greater than work split.")

        if self.needs_cache_block_size_2:
            if self.nx1 < self.cache_block_size_2:
                raise ValueError("cache_block_size_2 must be greater than work split.")

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

        self.vx = arguments.vx if arguments.vx is not None else np.zeros((self.ny1, self.nx1))
        self.vy = arguments.vy if arguments.vy is not None else np.zeros((self.ny1, self.nx1))
        self.vx_rhs  = arguments.vx_rhs if arguments.vx_rhs is not None else np.zeros((self.ny1, self.nx1))

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

        self.vy = arguments.vy if arguments.vy is not None else np.zeros((self.ny1, self.nx1))
        self.vx = arguments.vx if arguments.vx is not None else np.zeros((self.ny1, self.nx1))
        self.vy_rhs  = arguments.vy_rhs if arguments.vy_rhs is not None else np.zeros((self.ny1, self.nx1))

        self.vy_new = arguments.vy_new if arguments.vy_new is not None else None

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

        self.vx = arguments.vx if arguments.vx is not None else np.zeros((self.ny1, self.nx1))
        self.vx_rhs  = arguments.vx_rhs if arguments.vx_rhs is not None else np.zeros((self.ny1, self.nx1))

        self.vx_new = arguments.vx_new if arguments.vx_new is not None else None

        self.vy = arguments.vy if arguments.vy is not None else np.zeros((self.ny1, self.nx1))
        self.vy_rhs  = arguments.vy_rhs if arguments.vy_rhs is not None else np.zeros((self.ny1, self.nx1))

        self.vy_new = arguments.vy_new if arguments.vy_new is not None else None

        self.args = arguments