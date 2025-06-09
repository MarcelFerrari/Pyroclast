#!/usr/bin/python3

import argparse
import importlib
import itertools
import os
import warnings
from typing import Callable, Type, Optional

import numba as nb
from git import Repo

import benchmark.config as config
import benchmark.defaults as defaults
import benchmark.results_processing as res_proc
from Pyroclast.string_util import print_banner
from benchmark.benchmark_validators import (BenchmarkType, BenchmarkResults, BenchmarkRun,
                                            BenchmarkValidatorSmoother, BenchmarkValidatorVX, BenchmarkValidatorVY)
from benchmark.benchmark_wrapper import BenchmarkSmoother, BenchmarkVX, BenchmarkVY
from benchmark.utils import dtf

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--iterations",
                    default=defaults.max_iter,
                    type=int,
                    help=f"Number of iterations. Default: {defaults.max_iter}")
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
                    default=None,
                    type=str,
                    help="Modules to test")

# Testing Options
parser.add_argument("-p", "--profiling",
                    action="store_true",
                    help="Perform Profiling using pyinstrument")
parser.add_argument("-s", "--samples",
                    default=defaults.number_of_samples,
                    type=int,
                    help=f"Number of samples to generate. Default: {defaults.number_of_samples}")
parser.add_argument("-t", "--test",
                    default=defaults.types,
                    nargs="+",
                    type=BenchmarkType,
                    help=f"List of benchmark types to test (default is all, can be limited to only a subfunction). "
                         f"Default: {defaults.types}")
parser.add_argument("-a", "--cache_a",
                    default=[None],
                    type=int,
                    nargs="+",
                    help="Cache Block size. If benchmark requires cache block size and it is not provided, "
                         "an error will be raised")
parser.add_argument("-b", "--cache_b",
                    default=[None],
                    nargs="+",
                    type=int,
                    help="Secondary Cache Block size. If benchmark requires second cache block size and it is not provided, "
                         "an error will be raised")
parser.add_argument("-f", "--force",
                    action="store_true",
                    help="Force execution of benchmark with pending changes.")
parser.add_argument("-q", "--quiet",
                    action="store_true",
                    help="Suppress Warning")
parser.add_argument("-c", "--cpu",
                    type=int,
                    nargs="+",
                    default=None,
                    help=f"Number of CPU cores to use. Default: {os.cpu_count()}")
parser.add_argument("-e", "--no-env",
                    action="store_true",
                    help="Disable storing of environment variables.")
parser.add_argument("-P", "--print-table",
                    action="store_true",
                    help="Print result tables.")
parser.add_argument("-o", "--output",
                    type=str,
                    default=None,
                    help="Output file path or directory. If a directory is provided, "
                         "the file name generated is datetime(utc) + hash")
parser.add_argument("-l", "--list",
                    action="store_true",
                    help="List available benchmarks.")


def benchmark_single_module(module_name: str,
                            nx: int,
                            ny: int,
                            max_iter: int,
                            profiling: bool,
                            samples: int,
                            test_set: list[BenchmarkType],
                            cache_a: int,
                            cache_b: int,
                            cpu_count: int,
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

        print(f"Running Smoother Benchmark of: {module_name}")
        local_benchmark = bm_s(arguments=args)
        local_benchmark.benchmark()

        results.append(BenchmarkResults(
            module=module_name,
            benchmark_type=BenchmarkType.SMOOTHER,
            input_model=args,
            timings=local_benchmark.timings,
            cpu_count=cpu_count,
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

        print(f"Running VX Benchmark of: {module_name}")
        local_benchmark = bm_vx(arguments=args)
        local_benchmark.benchmark()

        results.append(BenchmarkResults(
            module=module_name,
            benchmark_type=BenchmarkType.VX,
            input_model=args,
            timings=local_benchmark.timings,
            cpu_count=cpu_count,
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

        print(f"Running VY Benchmark of: {module_name}")
        local_benchmark = bm_vy(arguments=args)
        local_benchmark.benchmark()

        results.append(BenchmarkResults(
            module=module_name,
            benchmark_type=BenchmarkType.VY,
            input_model=args,
            timings=local_benchmark.timings,
            cpu_count=cpu_count,
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


def get_git_info() -> tuple[str, str, str]:
    """
    Get branch and commit hash

    :returns: <branch name>, <commit hash>, <commit message>
    """
    repo = Repo(".", search_parent_directories=True)

    branch_name = repo.active_branch.name
    commit_hash = repo.active_branch.commit.hexsha
    commit_msg = repo.active_branch.commit.message

    return branch_name, commit_hash, commit_msg


def handle_store_run(run: BenchmarkRun, ns: argparse.Namespace):
    """
    Function implements the storage handling functionality.
    """
    # Parse output script parameter
    dest = None
    if ns.output is not None:
        dest = os.path.abspath(ns.output)

    # Attempt to get the config
    try:
        cfg = config.get_config()
    except FileNotFoundError:
        cfg = None

    # No output provided, config doesn't exist. -> Dump to stdout
    if dest is None:
        # Print warning
        if not ns.quiet and cfg is None:
            warnings.warn("No config file present. Writing benchmark data to stdout:")

        # Dump to stdout if config doesn't exist and now output is provided
        if cfg is None:
            print(run.model_dump_json())

        # Dump to directory specified in config
        else:
            # Store the run on the file system, using defaults
            res_proc.store_benchmark_run(run)
        return

    assert dest is not None, "INVARIANT: Destination must exist."

    # Get custom file name and custom direcory
    if os.path.isfile(dest):
        tgt_dir, file_name = os.path.split(dest)
    elif os.path.isdir(dest):
        tgt_dir = dest
        file_name = None
    else:
        if not ns.quiet:
            warnings.warn("Invalid Destination. Destination must be a directory or file. Dumping result to stdout")

        # Edge case of weird target (e.g. socket)
        print(run.model_dump_json())
        return

    # INFO: BenchmarkConfig correct, validate_hash_on_read has default.
    new_cfg = config.BenchmarkConfig(results_store=tgt_dir, day_folders=False, hash_suffix=True)

    # store with new config
    res_proc.store_benchmark_run(run, bmc=new_cfg, file_name=file_name)


def benchmark_lister() -> tuple[list[str], list[str], list[str]]:
    """
    List available implementations to benchmark. More specifically, walks through all files starting from
    Pyroclast/model/stokes_2D_mg/smoothers and checks, if they have a benchmark_factory and return a given benchmark

    :returns: vx benchmarks, vy benchmarks, smoother benchmarks
    """
    smoothers_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                                 "src", "Pyroclast", "model", "stokes_2D_mg", "smoothers"))

    vx_benchmarks = []
    vy_benchmarks = []
    smoother_benchmarks = []

    for path, dirs, files in os.walk(smoothers_dir):
        for file in files:
            # ignore non python files
            if not file.endswith(".py"):
                continue

            # ignore __init__ files
            if file == "__init__.py":
                continue

            # PRECONDITION: is python file, and is not init file
            file_path = os.path.join(path, file)

            # Remove till smooth
            mod_path = (file_path
                        .replace(smoothers_dir + "/", "" )
                        .replace(".py", "")
                        .replace("/", "."))

            module = importlib.import_module("Pyroclast.model.stokes_2D_mg.smoothers." + mod_path)

            if not hasattr(module, "benchmark_factory"):
                continue

            # PRECONDITION: is python file, is not init file, contains benchmark_factory
            smoother, vx, vy = getattr(module, "benchmark_factory")()

            if smoother is not None:
                smoother_benchmarks.append(mod_path)

            if vx is not None:
                vx_benchmarks.append(mod_path)

            if vy is not None:
                vy_benchmarks.append(mod_path)

    vx_benchmarks.sort()
    vy_benchmarks.sort()
    smoother_benchmarks.sort()

    return vx_benchmarks, vy_benchmarks, smoother_benchmarks


def main():
    """
    Main function to make it runnable from other locations.
    """
    print_banner()

    ns = parser.parse_args()

    print(ns)

    # Dim in x, y tuple
    dim_list: list[tuple[int, int]] = []
    all_res = []
    dirty = False

    branch, c_hash, c_msg = get_git_info()

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

    if ns.cpu is None:
        print(f"Process count not given, defaulting to os.cpu_count()={os.cpu_count()}")
        ns.cpu = [os.cpu_count()]

    # Check git status
    staged, unstaged = check_git_status()
    if staged or unstaged:
        if not ns.force:

            raise ValueError("Your working tree contains uncommited changes. Please commit or stash them. "
                             "By pass this guard with -f")
        else:
            dirty = True

    # Start of overall benchmark
    start = dtf()

    for ca in ns.cache_a:
        if len(ns.cache_a) > 1:
            print(f"Testing with Cache Size A = {ca}")

        for cb in ns.cache_b:
            if len(ns.cache_b) > 1:
                print(f"Testing with Cache Size B = {cb}")

            for cc in ns.cpu:
                # Set the cpu count
                nb.set_num_threads(cc)
                if len(ns.cpu) > 1:
                    print(f"Working on {cc} cpu's")

                # Run benchmark on modules and dimension list
                for module in ns.modules:
                    for dim in dim_list:
                        print(f"Running Module: {module} with dimensions: x={dim[0]}, y={dim[1]}")
                        all_res.extend(benchmark_single_module(module_name=module,
                                                               nx=dim[0], ny=dim[1], max_iter=ns.iterations,
                                                               profiling=ns.profiling, samples=ns.samples,
                                                               cache_a=ca, cache_b=cb,
                                                               test_set=ns.test, cpu_count=cc))

    # End of overall benchmark
    end = dtf()

    benchmark_run = BenchmarkRun(
        start=start, end=end,
        args=ns.__dict__,
        dirty=dirty, git_branch=branch, git_commit_hash=c_hash, git_commit_msg=c_msg,
        result=all_res,
        env=None if ns.no_env else os.environ,
    )

    # Print
    if ns.print_table:
        print(benchmark_run.model_dump_json(indent=2))

        print(f"Benchmark of {start.isoformat()}, time taken: {(end - start).total_seconds()}")
        res_proc.print_statistics(benchmark_run, False)

    handle_store_run(benchmark_run, ns)



if __name__ == "__main__":
    main()
