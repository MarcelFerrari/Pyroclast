#!/usr/bin/python3

import argparse
import importlib
import itertools
import os
import warnings
from typing import Callable, Type, Optional

from git import Repo

from benchmark.utils import dtf
from benchmark.benchmark_validators import (BenchmarkType, BenchmarkResults, BenchmarkRun,
                                            BenchmarkValidatorSmoother, BenchmarkValidatorVX, BenchmarkValidatorVY)
from benchmark.benchmark_wrapper import BenchmarkSmoother, BenchmarkVX, BenchmarkVY


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--iterations",
                    default=128,
                    type=int,
                    help="Number of iterations")
parser.add_argument("-d", "--dimension",
                    default=None,
                    type=int,
                    nargs="+",
                    help="List of domain sizes to try. DO NOT COMBINE WITH -x and -y")
parser.add_argument("-x", "--x-dimension",
                    default=None,
                    type=int,
                    nargs="+",
                    help="List of domain size in x dimension, tests cartesian product of x and y, "
                         "DO NOT COMBINE with -d")
parser.add_argument("-y", "--y-dimension",
                    default=None,
                    type=int,
                    nargs="+",
                    help="List of domain size in x dimension, tests cartesian product of x and y, "
                         "DO NOT COMBINE WITH -d")
parser.add_argument("-m", "--modules",
                    nargs="+",
                    type=str,
                    required=True,
                    help="Modules to test")

# Testing Options
parser.add_argument("-p", "--profiling",
                    action="store_true",
                    help="Perform Profiling using pyinstrument")
parser.add_argument("-s", "--samples",
                    default=1,
                    type=int,
                    help="Number of samples to generate")
parser.add_argument("-t", "--test",
                    default=[BenchmarkType.SMOOTHER, BenchmarkType.VX, BenchmarkType.VY],
                    nargs="+",
                    type=BenchmarkType,
                    help="List of benchmark types to test (default is all, can be limited to only a subfunction)")
parser.add_argument("-a", "--cache_a",
                    default=None,
                    type=int,
                    help="Cache Block size. If benchmark requires cache block size and it is not provided, "
                         "an error will be raised")
parser.add_argument("-b", "--cache_b",
                    default=None,
                    type=int,
                    help="Secondary Cache Block size. If benchmark requires second cache block size and it is not provided, "
                         "an error will be raised")
parser.add_argument("-f", "--force",
                    action="store_true",
                    help="Force execution of benchmark with pending changes.")
parser.add_argument("-q", "--quiet",
                    action="store_true",
                    help="Suppress warnings")
parser.add_argument("-c", "--cpu",
                    type=int,
                    default=None,
                    help="Number of CPU cores to use")
parser.add_argument(f"-e", "--no-env",
                    action="store_false",
                    help="Disable storing of environment variables.")


def benchmark_single_module(module_name: str,
                            nx: int,
                            ny: int,
                            max_iter: int,
                            profiling: bool,
                            samples: int,
                            test_set: list[BenchmarkType],
                            cache_a: int,
                            cache_b: int,
                            ) -> list[BenchmarkResults]:
    """
    Run the benchmark for a single module. Requires this module to have defined a benchmark factory.
    """
    results = []
    module = importlib.import_module(f"Pyroclast.model.stokes_2D_mg.smoothers.{module_name}")

    # Get the factory method
    try:
        factory = getattr(module, "benchmark_factory")
    except AttributeError:
        raise RuntimeError(f"Module {module_name} does not implement benchmark_factory. "
                           f"The targeted smoother implements benchmark_factory and the return type is correct. "
                           f"Check the template.py in the `smoothers` package for the signature of the factory")

    # Annotated factory
    factory: Callable[[], tuple[Optional[Type[BenchmarkSmoother]],
                                Optional[Type[BenchmarkVX]],
                                Optional[Type[BenchmarkVY]]]]

    # Execute factory
    bm_s, bm_vx, bm_vy = factory()

    # Run benchmark for entire smoother
    if bm_s is not None and BenchmarkType.SMOOTHER in test_set:
        args = BenchmarkValidatorSmoother(
            nx=nx, ny=ny,
            max_iter=max_iter,
            profile=profiling, samples=samples,
            cache_block_size_1=cache_a,
            cache_block_size_2=cache_b,
        )

        local_benchmark = bm_s(arguments=args)
        local_benchmark.benchmark()

        results.append(BenchmarkResults(
            module=module_name,
            benchmark_type=BenchmarkType.SMOOTHER,
            input_model=args,
            timings=local_benchmark.timings
        ))

    # Run benchmark on vx_subroutine
    if bm_vx is not None and BenchmarkType.VX in test_set:
        args = BenchmarkValidatorVX(
            nx=nx, ny=ny,
            max_iter=max_iter,
            profile=profiling, samples=samples,
            cache_block_size_1=cache_a,
            cache_block_size_2=cache_b,
        )

        local_benchmark = bm_vx(arguments=args)
        local_benchmark.benchmark()

        results.append(BenchmarkResults(
            module=module_name,
            benchmark_type=BenchmarkType.VX,
            input_model=args,
            timings=local_benchmark.timings
        ))

    # Run benchmark on vy_subroutine
    if bm_vy is not None and BenchmarkType.VY in test_set:
        args = BenchmarkValidatorVY(
            nx=nx, ny=ny,
            max_iter=max_iter,
            profile=profiling, samples=samples,
            cache_block_size_1=cache_a,
            cache_block_size_2=cache_b,
        )

        local_benchmark = bm_vy(arguments=args)
        local_benchmark.benchmark()

        results.append(BenchmarkResults(
            module=module_name,
            benchmark_type=BenchmarkType.VY,
            input_model=args,
            timings=local_benchmark.timings
        ))

    return results


def check_git_status() -> tuple[bool, bool]:
    """
    Check the git status and return True, if the working tree is clean

    returns: [has_staged_changes, has_unstaged_changes]
    """
    repo = Repo(".", search_parent_directories=True)

    # 1. Check for staged changes (ready to commit)
    staged = repo.index.diff("HEAD")  # staged vs HEAD
    has_staged_changes = bool(staged)

    # 2. Check for unstaged changes (working tree vs index)
    unstaged = repo.index.diff(None)  # index vs working tree
    has_unstaged_changes = bool(unstaged)

    return has_staged_changes, has_unstaged_changes


def main():
    """
    Main function to make it runnable from other locations.
    """
    ns = parser.parse_args()
    # Dim in x, y tuple
    dim_list: list[tuple[int, int]] = []
    all_res = []
    dirty = False

    print(ns.__dict__)

    # Prevent all empty
    if ns.dimension is None and ns.x_dimension is None and ns.y_dimension is None:
        raise ValueError("At least one kind of dimension needs to be provided "
                         "(either dimension or x-dimensions and y-dimensions)")

    # dimension is given
    if ns.dimension is not None:
        if ns.x_dimension is not None or ns.y_dimension is not None:
            raise ValueError("x-dimension and y-dimension need to be empty if dimension is provided")

        dim_list = [(d, d) for d in ns.dimension]

    # x and y are given
    if ns.x_dimension is not None and ns.y_dimension is not None:
        if ns.dimension is not None:
            raise ValueError("dimension needs to be empty if x-dimension and y-dimension are provided")

        for x, y in itertools.product(ns.x_dimension, ns.y_dimension):
            dim_list.append((x, y))

    # Warn that there's an issue with the configuration.
    if ns.samples > 1 and ns.profiling and not ns.quiet:
        warnings.warn("Profiling in Combination with multiple samples increases runtime drastically")

    # Check git status
    staged, unstaged = check_git_status()
    if staged or unstaged:
        if not ns.force:

            raise ValueError("Your working tree contains uncommited changes. Please commit or stash them. "
                             "By pass this guard with -f")
        else:
            dirty = True

    # Run benchmark on modules and dimension list
    for module in ns.modules:
        for dim in dim_list:
            all_res.extend(benchmark_single_module(module=module,
                                                   nx=dim[0], ny=dim[1], max_iter=ns.iterations,
                                                   profiling=ns.profiling, sample=ns.samples,
                                                   cache_a=ns.cache_a, cache_b=ns.cache_b,
                                                   test_set=ns.test))

if __name__ == "__main__":
    main()
