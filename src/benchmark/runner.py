#!/usr/bin/python3

"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: benchmark/runner.py 
Description: File contains functionality to run reproducible benchmarks of the smoothers as part of 
             Alexander Sotoudeh's Bachelor's Thesis

Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""     

import argparse
import importlib
import importlib.metadata
import itertools
import os
import warnings
from typing import Callable, Type, Optional

import numba as nb

import benchmark.config as config
import benchmark.defaults as defaults
import benchmark.results_processing as res_proc
from Pyroclast.string_util import print_banner
from benchmark.benchmark_validators import (BenchmarkType, BenchmarkResults, BenchmarkRun,
                                            BenchmarkValidatorSmoother, BenchmarkValidatorVX, BenchmarkValidatorVY)
from benchmark.benchmark_wrapper import BenchmarkSmoother, BenchmarkVX, BenchmarkVY
from benchmark.git_checks import get_git_info, check_git_status
from benchmark.utils import dtf

parser = argparse.ArgumentParser(prog="Smoother Benchmark Runner",
                                 description="Run benchmarks on the smoother implementation of Pyroclast. Both full "
                                             "smoother benchmarks, as well as partial benchmarks only on the vx or vy "
                                             "component can be run")
perf_opt = parser.add_argument_group(
    title="Performance Options",
    description="Variables which have a major impact on performance. "
                "All variables accept >= 1 arguments. The benchmarking algorithm will test the "
                "cartesian product of all configurations. The benchmark will warn you, if you need to provide another "
                "performance option (like caches) for the benchmark to run successfully. However, it will not inform "
                "you, if a given benchmark is not influenced by a given variable")

bench_opt = parser.add_argument_group(
    title="Benchmark Options",
    description="Variables which determine what the benchmark does. E.g. storage path for results, printing of results,"
                "optional stack analysis with pyinstrument... "
)
perf_opt.add_argument("-i", "--iterations",
                      default=defaults.max_iter,
                      type=int,
                      help=f"Number of iterations. Default: {defaults.max_iter}")
perf_opt.add_argument("-d", "--dimension",
                      default=None,
                      type=int,
                      nargs="+",
                      help="List of domain sizes to try. DO NOT COMBINE WITH -x and -y")
perf_opt.add_argument("-x", "--x-dimension",
                      default=None,
                      type=int,
                      nargs="+",
                      help="List of domain size in x dimension, tests cartesian product of x and y, "
                           "DO NOT COMBINE with -d")
perf_opt.add_argument("-y", "--y-dimension",
                      default=None,
                      type=int,
                      nargs="+",
                      help="List of domain size in x dimension, tests cartesian product of x and y, "
                           "DO NOT COMBINE WITH -d")
perf_opt.add_argument("-m", "--modules",
                      nargs="+",
                      default=None,
                      type=str,
                      help="Modules to test")
perf_opt.add_argument("-a", "--cache_a",
                      default=[None],
                      type=int,
                      nargs="+",
                      help="Cache Block size. If benchmark requires cache block size and it is not provided, "
                           "an error will be raised. If the benchmark doesn't require a cache_a option, "
                           "it will be ignored.")
perf_opt.add_argument("-b", "--cache_b",
                      default=[None],
                      nargs="+",
                      type=int,
                      help="Secondary Cache Block size. If benchmark requires second cache block size and it is not "
                           "provided, an error will be raised. If the benchmark doesn't require a cache_b option, "
                           "it will be ignored.")
perf_opt.add_argument("-c", "--cpu",
                      type=int,
                      nargs="+",
                      default=None,
                      help=f"Number of CPU cores to use. Default: {os.cpu_count()}")
perf_opt.add_argument("-u", "--unroll",
                      type=int,
                      nargs="+",
                      default=[4],
                      help=f"Number of Loop unrolls to do with second layer caching")
perf_opt.add_argument("-j", "--jitter",
                      type=int,
                      nargs="+",
                      default=[3],
                      help=f"Amount of boundary jitter in cells. Given 3 jitter is +/- 3")

# Testing Options
bench_opt.add_argument("-p", "--profiling",
                       action="store_true",
                       help="Perform Profiling using pyinstrument")
bench_opt.add_argument("-s", "--samples",
                       default=defaults.number_of_samples,
                       type=int,
                       help=f"Number of samples to generate. Default: {defaults.number_of_samples}")
bench_opt.add_argument("-t", "--test",
                       default=defaults.benchmark_types,
                       nargs="+",
                       type=BenchmarkType,
                       help=f"List of benchmark types to test (default is all, can be limited to only a subset). "
                            f"Default: {list(map(lambda t: t.value, defaults.benchmark_types))}")
bench_opt.add_argument("-f", "--force",
                       action="store_true",
                       help="Force execution of benchmark with pending changes.")
bench_opt.add_argument("-q", "--quiet",
                       action="store_true",
                       help="Suppress Warning")
bench_opt.add_argument("-e", "--no-env",
                       action="store_true",
                       help="Disable storing of environment variables.")
bench_opt.add_argument("-P", "--print-table",
                       action="store_true",
                       help="Print result tables.")
bench_opt.add_argument("-o", "--output",
                       type=str,
                       default=None,
                       help="Output file path or directory. If a directory is provided, "
                            "the file name generated is datetime(utc) + hash")
bench_opt.add_argument("-l", "--list",
                       action="store_true",
                       help="List available benchmarks.")
bench_opt.add_argument(f"-B", "--no-burn-in",
                       action="store_true",
                       help="Skip Burn-In Phase base_rb_gb.vx benchmark for 60s to preheat CPU.")


# ======================================================================================================================
# Functions used to perform main benchmarking operations
# ======================================================================================================================


def burn_in_cpu():
    """
    Run base_rb_gs vx routine for a given amount of time to preheat the cpu.
    """
    module = importlib.import_module(f"Pyroclast.solvers.stokes_2d.smoothers.base_rb_gs")
    factory = getattr(module, "benchmark_factory")
    # Annotated factory
    factory: Callable[[], tuple[Optional[Type[BenchmarkSmoother]],
                                Optional[Type[BenchmarkVX]],
                                Optional[Type[BenchmarkVY]]]]

    # Execute factory
    bm_s, bm_vx, bm_vy = factory()

    # Attempt to get the config
    try:
        cfg = config.get_config()
    except FileNotFoundError:
        cfg = None

    timeout = cfg.burn_in_timeout if cfg is not None else 60

    args = BenchmarkValidatorVX(
        nx=1024, ny=1024,
        max_iter=128,
        profile=False, samples=15,
        cache_block_size_1=None,
        cache_block_size_2=None,
    )

    print(f"Starting CPU Burn-In")

    # Do burn in
    start = dtf()
    while (dtf() - start).total_seconds() < timeout:
        print(f"{timeout - (dtf() - start).total_seconds()} seconds remaining")
        bm_vx(args).benchmark()

    print(f"Burn in for {timeout} seconds done.")


NUMBA_CUDA_AVAIL = False


try:
    importlib.metadata.version("numba-cuda")
    NUMBA_CUDA_AVAIL = True
except importlib.metadata.PackageNotFoundError:
    pass


# If Numba Cuda is available, prepare the decorators for the gpu exports as well.
if NUMBA_CUDA_AVAIL:
    def burn_in_gpu():
        """
        Run base_rb_gs vx routine for a given amount of time to preheat the cpu.
        """
        module = importlib.import_module(f"Pyroclast.solvers.stokes_2d.smoothers.gpu_jacobi")
        factory = getattr(module, "benchmark_factory")

        # Annotated factory
        factory: Callable[[], tuple[Optional[Type[BenchmarkSmoother]],
                                    Optional[Type[BenchmarkVX]],
                                    Optional[Type[BenchmarkVY]]]]

        # Execute factory
        bm_s, bm_vx, bm_vy = factory()

        # Attempt to get the config
        try:
            cfg = config.get_config()
        except FileNotFoundError:
            cfg = None

        timeout = cfg.burn_in_timeout if cfg is not None else 60

        args = BenchmarkValidatorSmoother(
            nx=1024, ny=1024,
            max_iter=128,
            profile=False, samples=1,
            cache_block_size_1=None,
            cache_block_size_2=None,
            iter_unroll=None
        )

        print(f"Starting GPU Burn-In")

        # Do burn in
        start = dtf()
        while (dtf() - start).total_seconds() < timeout:
            print(f"{timeout - (dtf() - start).total_seconds()} seconds remaining")
            bm_s(args).benchmark()

        print(f"Burn in for {timeout} seconds done.")
else:
    raise ImportError("numba-cuda needed for gpu support. Try installing Pyroclast[cuda]")


def benchmark_smoother(nx: int, ny: int,
                       max_iter: int,
                       profiling: bool, samples: int,
                       cache_block_size_1: int, cache_block_size_2: int,
                       module_name: str,
                       is_gpu: bool,
                       cpu_count: int,
                       iter_unroll: int, jitter: int,
                       benchmark: Type[BenchmarkSmoother]) -> BenchmarkResults:
    """
    Run the smoother benchmark
    """
    args = BenchmarkValidatorSmoother(
        nx=nx, ny=ny,
        max_iter=max_iter,
        profile=profiling, samples=samples,
        cache_block_size_1=cache_block_size_1,
        cache_block_size_2=cache_block_size_2,
        iter_unroll=iter_unroll,
        jitter=jitter,
    )

    ca_str = f"Cache Size 1: {cache_block_size_1}" if benchmark.needs_cache_block_size_1 else ""
    cb_str = f"Cache Size 2: {cache_block_size_2}" if benchmark.needs_cache_block_size_2 else ""
    iu_str = f"Iter Unroll: {iter_unroll}" if benchmark.needs_iter_unroll else ""

    print(f"Running Smoother Benchmark of: {module_name}\n"
          f"Dimension: {nx} x {ny}\nCPUs: {cpu_count}\n{ca_str}\n{cb_str}\n{iu_str}")
    local_benchmark = benchmark(arguments=args)
    local_benchmark.benchmark()

    return BenchmarkResults(
        module=module_name,
        benchmark_type=BenchmarkType.SMOOTHER,
        input_model=args,
        timings=local_benchmark.timings,
        cpu_count=cpu_count,
        is_gpu=is_gpu
    )


def benchmark_vx(nx: int, ny: int,
                 max_iter: int,
                 profiling: bool, samples: int,
                 cache_block_size_1: int, cache_block_size_2: int,
                 module_name: str,
                 is_gpu: bool,
                 cpu_count: int,
                 iter_unroll: int,
                 jitter: int,
                 benchmark: Type[BenchmarkVX]) -> BenchmarkResults:
    """
    Run the smoother benchmark
    """
    args = BenchmarkValidatorVX(
        nx=nx, ny=ny,
        max_iter=max_iter,
        profile=profiling, samples=samples,
        cache_block_size_1=cache_block_size_1,
        cache_block_size_2=cache_block_size_2,
        iter_unroll=iter_unroll,
        jitter=jitter,
    )

    ca_str = f"Cache Size 1: {cache_block_size_1}" if benchmark.needs_cache_block_size_1 else ""
    cb_str = f"Cache Size 2: {cache_block_size_2}" if benchmark.needs_cache_block_size_2 else ""
    iu_str = f"Iter Unroll: {iter_unroll}" if benchmark.needs_iter_unroll else ""

    print(f"Running Benchmark VX of: {module_name}\n"
          f"Dimension: {nx} x {ny}\nCPUs: {cpu_count}\n{ca_str}\n{cb_str}\n{iu_str}")
    local_benchmark = benchmark(arguments=args)
    local_benchmark.benchmark()

    return BenchmarkResults(
        module=module_name,
        benchmark_type=BenchmarkType.SMOOTHER,
        input_model=args,
        timings=local_benchmark.timings,
        cpu_count=cpu_count,
        is_gpu=is_gpu
    )


def benchmark_vy(nx: int, ny: int,
                 max_iter: int,
                 profiling: bool, samples: int,
                 cache_block_size_1: int, cache_block_size_2: int,
                 module_name: str,
                 is_gpu: bool,
                 cpu_count: int,
                 iter_unroll: int,
                 jitter: int,
                 benchmark: Type[BenchmarkVY]) -> BenchmarkResults:
    """
    Run the smoother benchmark
    """
    args = BenchmarkValidatorVY(
        nx=nx, ny=ny,
        max_iter=max_iter,
        profile=profiling, samples=samples,
        cache_block_size_1=cache_block_size_1,
        cache_block_size_2=cache_block_size_2,
        iter_unroll=iter_unroll,
        jitter=jitter,
    )

    ca_str = f"Cache Size 1: {cache_block_size_1}" if benchmark.needs_cache_block_size_1 else ""
    cb_str = f"Cache Size 2: {cache_block_size_2}" if benchmark.needs_cache_block_size_2 else ""
    iu_str = f"Iter Unroll: {iter_unroll}" if benchmark.needs_iter_unroll else ""

    print(f"Running Benchmark VY of: {module_name}\n"
          f"Dimension: {nx} x {ny}\nCPUs: {cpu_count}\n{ca_str}\n{cb_str}\n{iu_str}")
    local_benchmark = benchmark(arguments=args)
    local_benchmark.benchmark()

    return BenchmarkResults(
        module=module_name,
        benchmark_type=BenchmarkType.SMOOTHER,
        input_model=args,
        timings=local_benchmark.timings,
        cpu_count=cpu_count,
        is_gpu=is_gpu
    )


def benchmark_single_module(module_name: str,
                            dim_list: list[tuple[int, int]],
                            max_iter: int,
                            profiling: bool,
                            samples: int,
                            test_set: list[BenchmarkType],
                            cache_a: list[int],
                            cache_b: list[int],
                            cpu_count: list[int],
                            iter_unroll = list[int],
                            jitter=list[int],
                            ) -> list[BenchmarkResults]:
    """
    Run the benchmark for a single module. Requires this module to have defined a benchmark factory.
    """
    results = []
    module = importlib.import_module(f"Pyroclast.solvers.stokes_2d.smoothers.{module_name}")

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

    is_gpu = getattr(module, "IS_GPU", False)

    # Execute factory
    bm_s, bm_vx, bm_vy = factory()

    # Run benchmark for entire smoother
    if bm_s is not None and BenchmarkType.SMOOTHER in test_set:
        caches_a_loc = cache_a.copy() if bm_s.needs_cache_block_size_1 else [None]
        caches_b_loc = cache_b.copy() if bm_s.needs_cache_block_size_2 else [None]
        local_iur = iter_unroll.copy() if bm_s.needs_iter_unroll else [None]
        jitter_loc = jitter.copy() if bm_s.needs_jitter else [None]

        for ca, cb, dim, cc, iu, j in itertools.product(sorted(caches_a_loc),
                                                     sorted(caches_b_loc),
                                                     sorted(dim_list, reverse=True, key=lambda d: d[0] * d[1]),
                                                     sorted(cpu_count, reverse=True),
                                                     sorted(local_iur, reverse=True),
                                                        sorted(jitter_loc)):
            print(cc)
            nb.set_num_threads(cc)
            results.append(benchmark_smoother(nx=dim[0], ny=dim[1],
                                              max_iter=max_iter,
                                              profiling=profiling, samples=samples,
                                              cache_block_size_1=ca, cache_block_size_2=cb,
                                              module_name=module_name,
                                              is_gpu=is_gpu,
                                              cpu_count=cc,
                                              benchmark=bm_s,
                                              iter_unroll=iu,
                                              jitter=j))

    # Run benchmark on vx_subroutine
    if bm_vx is not None and BenchmarkType.VX in test_set:
        caches_a_loc = cache_a.copy() if bm_vx.needs_cache_block_size_1 else [None]
        caches_b_loc = cache_b.copy() if bm_vx.needs_cache_block_size_2 else [None]
        local_iur = iter_unroll.copy() if bm_vx.needs_iter_unroll else [None]
        jitter_loc = jitter.copy() if bm_vx.needs_jitter else [None]

        for ca, cb, dim, cc, iu, j in itertools.product(sorted(caches_a_loc),
                                                     sorted(caches_b_loc),
                                                     sorted(dim_list, reverse=True, key=lambda d: d[0] * d[1]),
                                                     sorted(cpu_count, reverse=True),
                                                     sorted(local_iur, reverse=True),
                                                     sorted(jitter_loc)):

            nb.set_num_threads(cc)
            results.append(benchmark_vx(nx=dim[0], ny=dim[1],
                                        max_iter=max_iter,
                                        profiling=profiling, samples=samples,
                                        cache_block_size_1=ca, cache_block_size_2=cb,
                                        module_name=module_name,
                                        is_gpu=is_gpu,
                                        cpu_count=cc,
                                        benchmark=bm_vx,
                                        iter_unroll=iu,
                                        jitter=j))

    # Run benchmark on vy_subroutine
    if bm_vy is not None and BenchmarkType.VY in test_set:
        caches_a_loc = cache_a.copy() if bm_vy.needs_cache_block_size_1 else [None]
        caches_b_loc = cache_b.copy() if bm_vy.needs_cache_block_size_2 else [None]
        local_iur = iter_unroll.copy() if bm_vy.needs_iter_unroll else [None]
        jitter_loc = jitter.copy() if bm_vy.needs_jitter else [None]

        for ca, cb, dim, cc, iu, j in itertools.product(sorted(caches_a_loc),
                                                     sorted(caches_b_loc),
                                                     sorted(dim_list, reverse=True, key=lambda d: d[0] * d[1]),
                                                     sorted(cpu_count, reverse=True),
                                                     sorted(local_iur, reverse=True),
                                                     sorted(jitter_loc)):

            nb.set_num_threads(cc)
            results.append(benchmark_vy(nx=dim[0], ny=dim[1],
                                        max_iter=max_iter,
                                        profiling=profiling, samples=samples,
                                        cache_block_size_1=ca, cache_block_size_2=cb,
                                        module_name=module_name,
                                        is_gpu=is_gpu,
                                        cpu_count=cc,
                                        benchmark=bm_vy,
                                        iter_unroll=iu,
                                        jitter=j))

    return results


def handle_store_run(run: BenchmarkRun, arg_dict: dict) -> Optional[str]:
    """
    Function implements the storage handling functionality.

    :returns: path to benchmark json file if successful, None otherwise
    """
    # Parse output script parameter
    dest = None
    if arg_dict["output"] is not None:
        dest = os.path.abspath(arg_dict["output"])

    # Attempt to get the config
    try:
        cfg = config.get_config()
    except FileNotFoundError:
        cfg = None

    # No output provided, config doesn't exist. -> Dump to stdout
    if dest is None:
        # Print warning
        if not arg_dict["quiet"] and cfg is None:
            warnings.warn("No config file present. Writing benchmark data to stdout:")

        # Dump to stdout if config doesn't exist and now output is provided
        if cfg is None:
            print(run.model_dump_json())
            return None

        # Dump to directory specified in config
        else:
            # Store the run on the file system, using defaults
            return res_proc.store_benchmark_run(run)

    assert dest is not None, "INVARIANT: Destination must exist."

    # Get custom file name and custom direcory
    if os.path.isfile(dest):
        tgt_dir, file_name = os.path.split(dest)
    elif os.path.isdir(dest):
        tgt_dir = dest
        file_name = None
    else:
        if not arg_dict["quiet"]:
            warnings.warn("Invalid Destination. Destination must be a directory or file. Dumping result to stdout")

        # Edge case of weird target (e.g. socket)
        print(run.model_dump_json())
        return None

    # INFO: BenchmarkConfig correct, validate_hash_on_read has default.
    new_cfg = config.BenchmarkConfig(results_store=tgt_dir, day_folders=False, hash_suffix=True, plot_store=".")

    # store with new config
    return res_proc.store_benchmark_run(run, bmc=new_cfg, file_name=file_name)


def benchmark_lister() -> tuple[list[str], list[str], list[str]]:
    """
    List available implementations to benchmark. More specifically, walks through all files starting from
    Pyroclast/model/stokes_2D_mg/smoothers and checks, if they have a benchmark_factory and return a given benchmark

    :returns: vx benchmarks, vy benchmarks, smoother benchmarks
    """
    smoothers_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                                 "src", "Pyroclast", "solvers", "stokes_2d", "smoothers"))

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

            module = importlib.import_module("Pyroclast.solvers.stokes_2d.smoothers." + mod_path)

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


def partition_benchmark(modules: list[str]) -> tuple[list[str], list[str]]:
    """
    Partition the benchmarks into cpu benchmarks and gpu benchmarks.
    """
    cpu_modules = []
    gpu_modules = []

    for module_path in modules:
        module = importlib.import_module("Pyroclast.solvers.stokes_2d.smoothers." + module_path)

        if not hasattr(module, "benchmark_factory"):
            continue

        if getattr(module, "IS_GPU", False):
            gpu_modules.append(module_path)
        else:
            cpu_modules.append(module_path)

    cpu_modules.sort()
    gpu_modules.sort()

    return cpu_modules, gpu_modules


# ======================================================================================================================
# Main function which calls all other functions (needed to expose this command)
# ======================================================================================================================

def perform_benchmark_run(arg_dict: dict):
    """
    Function contains everything necessary to perform a full benchmark run
    """
    # Check arg dict contains the necessary keys
    assert set(arg_dict.keys()) == {
        "iterations",
        "dimension",
        "x_dimension",
        "y_dimension",
        "modules",
        "cache_a",
        "cache_b",
        "cpu",
        "profiling",
        "samples",
        "test",
        "force",
        "quiet",
        "no_env",
        "print_table",
        "output",
        "list",
        "no_burn_in",
        "unroll",
        "jitter"
    }, "INCORRECT ARGUMENTS. Check arguments provided match expected arguments dict."

    # Dim in x, y tuple
    dim_list: list[tuple[int, int]] = []
    all_res = []
    dirty = False

    # Get the git information
    branch, c_hash, c_msg = get_git_info()

    # Validate iter unroll
    for unroll in arg_dict["unroll"]:
        if unroll is not None and unroll > arg_dict["iterations"]:
            raise ValueError(f"Cannot unroll more loops than number of iterations"
                             f" {unroll} of loops to unroll with {arg_dict['iterations']}")

    # Only list benchmarks
    if arg_dict["list"] is True:
        vx, vy, smoother = benchmark_lister()
        print(f"VX Benchmarks:\n" + "\n".join(vx) + "\n")
        print(f"VY Benchmarks:\n" + "\n".join(vy) + "\n")
        print(f"Smoother Benchmarks:\n" + "\n".join(smoother) + "\n")
        return

    if arg_dict["modules"] is None:
        raise ValueError("At least one Module is required for benchmarking.")

    # dimension is given
    if arg_dict["dimension"] is not None and arg_dict["x_dimension"] is None or arg_dict["y_dimension"] is None:
        dim_list = [(d, d) for d in arg_dict["dimension"]]

    # x and y are given
    elif arg_dict["x_dimension"] is not None and arg_dict["y_dimension"] is not None and arg_dict["dimension"] is None:
        for x, y in itertools.product(arg_dict["x_dimension"], arg_dict["y_dimension"]):
            dim_list.append((x, y))
    else:
        raise ValueError("Invalid Dimension specification. Either provide -d <list of dimensions> or "
                         "-x <list of x sizes> and -y <list of y sizes>")

    # Warn that there's an issue with the configuration.
    if arg_dict["samples"] > 1 and arg_dict["profiling"] and not arg_dict["quiet"]:
        warnings.warn("Profiling in Combination with multiple samples increases runtime drastically")

    if arg_dict["cpu"] is None:
        print(f"Process count not given, defaulting to os.cpu_count()={os.cpu_count()}")
        arg_dict["cpu"] = [os.cpu_count()]

    # Check git status
    staged, unstaged = check_git_status()
    if staged or unstaged:
        if not arg_dict["force"]:

            raise ValueError("Your working tree contains uncommited changes. Please commit or stash them. "
                             "By pass this guard with -f")
        else:
            dirty = True

    nb.config.THREADING_LAYER = "omp"

    cpu_modules, gpu_modules = partition_benchmark(arg_dict["modules"])

    # Start of overall benchmark
    start = dtf()

    # Run CPU Benchmarks
    if len(cpu_modules) > 0:
        if not arg_dict["no_burn_in"]:
            burn_in_cpu()

        # Run benchmark on modules and dimension list
        for module in sorted(cpu_modules):
            all_res.extend(benchmark_single_module(
                module_name=module,
                dim_list=dim_list, max_iter=arg_dict["iterations"],
                profiling=arg_dict["profiling"], samples=arg_dict["samples"],
                cache_a=arg_dict["cache_a"], cache_b=arg_dict["cache_b"], iter_unroll=arg_dict["unroll"],
                test_set=arg_dict["test"], cpu_count=arg_dict["cpu"], jitter=arg_dict["jitter"]))

    # Run GPU benchmarks
    if len(gpu_modules) > 0:
        if not NUMBA_CUDA_AVAIL:
            raise ImportError("numba-cuda needed for gpu support. Try installing Pyroclast[cuda]")

        if not arg_dict["no_burn_in"]:
            burn_in_gpu()

        # Run benchmark on modules and dimension list
        for module in sorted(gpu_modules):
            all_res.extend(benchmark_single_module(
                module_name=module,
                dim_list=dim_list, max_iter=arg_dict["iterations"],
                profiling=arg_dict["profiling"], samples=arg_dict["samples"],
                cache_a=arg_dict["cache_a"], cache_b=arg_dict["cache_b"], iter_unroll=arg_dict["unroll"],
                test_set=arg_dict["test"], cpu_count=arg_dict["cpu"], jitter=arg_dict["jitter"]))

    # End of overall benchmark
    end = dtf()

    benchmark_run = BenchmarkRun(
        start=start, end=end,
        args=arg_dict,
        dirty=dirty, git_branch=branch, git_commit_hash=c_hash, git_commit_msg=c_msg,
        result=all_res,
        env=None if arg_dict["no_env"] else os.environ,
    )

    # Print
    if arg_dict["print_table"]:
        print(benchmark_run.model_dump_json(indent=2))

        print(f"Benchmark of {start.isoformat()}, time taken: {(end - start).total_seconds()}")
        res_proc.print_statistics(benchmark_run, False)
        print(f"Benchmarking took: {(end - start).total_seconds()}s")

    handle_store_run(benchmark_run, arg_dict)


def main():
    """
    Main function to make it runnable from other locations.
    """
    print_banner()

    ns = parser.parse_args()

    arg_dict = vars(ns)
    perform_benchmark_run(arg_dict)


if __name__ == "__main__":
    main()
