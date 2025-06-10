import base64
import hashlib
import os.path
from typing import Any
from typing import Optional

import copy
import numpy as np
import pandas as pd
import tabulate

from benchmark.benchmark_validators import BenchmarkResults, BenchmarkRun, Stage, Timing
from benchmark.config import get_config, BenchmarkConfig


"""
This file contains some functionality to handle formatting and printing of benchmark results.
"""


def extract_table_data(benchmarks: list[BenchmarkResults], add_debug: bool = False):
    """
    Extract dicts to be printed with tabulate
    """
    formated_benchmarks = []
    for b in benchmarks:
        base_dict: dict[str, Any] = {
            "Module": b.module,
            "cpu_count": b.cpu_count,
            "nx": b.input_model.nx,
            "ny": b.input_model.ny,
        }
        norm_div = b.input_model.nx * b.input_model.ny * b.input_model.max_iter

        if b.input_model.samples == 1:
            base_dict.update({"runtime": b.benchmark_timings[0].duration,
                              "runtime normalized": b.benchmark_timings[0].duration / norm_div})
        else:
            durations = [t.duration for t in b.benchmark_timings]
            base_dict.update({"runtime std:": float(np.std(durations)),
                              "runtime avg": float(np.mean(durations)),
                              "runtime std norm:": float(np.std(durations)) / norm_div,
                              "runtime avg norm": float(np.mean(durations)) / norm_div,
                              "number of samples": b.input_model.samples
                              })

        if add_debug:
            base_dict.update({"preamble time": b.preamble_timing.duration if b.preamble_timing is not None else 0,
                              "epilog time": b.epilog_timing.duration if b.epilog_timing is not None else 0})

        formated_benchmarks.append(base_dict)

    return formated_benchmarks


def print_statistics(run: BenchmarkRun, add_debug: bool = False):
    """
    For each benchmark type, extract the statistics and print them

    :param run: BenchmarkRun (Benchmark Run to take and format)
    :param add_debug: Add debug info (preamble run time and epilog timing)
    """
    vx = run.vx_benchmarks
    vy = run.vy_benchmarks
    smoother = run.smoother_benchmarks

    # Sorted
    vx.sort(key=lambda v: (v.module, v.cpu_count, v.input_model.nx, v.input_model.ny))
    vy.sort(key=lambda v: (v.module, v.cpu_count, v.input_model.nx, v.input_model.ny))
    smoother.sort(key=lambda v: (v.module, v.cpu_count, v.input_model.nx, v.input_model.ny))

    formatted_vx = extract_table_data(vx, add_debug=add_debug)
    formatted_vy = extract_table_data(vy, add_debug=add_debug)
    formatted_smoother = extract_table_data(smoother, add_debug=add_debug)

    if len(formatted_vx) > 0:
        print(f"Benchmark Results of VX Benchmark")
        print(tabulate.tabulate(formatted_vx, headers="keys"))
        print()

    if len(formatted_vy) > 0:
        print(f"Benchmark Results of VY Benchmark")
        print(tabulate.tabulate(formatted_vy, headers="keys"))
        print()

    if len(formatted_smoother) > 0:
        print(f"Benchmark Results of Smoother Benchmark")
        print(tabulate.tabulate(formatted_smoother, headers="keys"))
        print()

    return formatted_vx, formatted_vy, formatted_smoother


def benchmark_run_string_hasher(benchmark: BenchmarkRun | str) -> str:
    """
    Compute the hash of the serialized benchmark result.
    """
    if isinstance(benchmark, str):
        hash_result = hashlib.sha256(benchmark.encode("utf-8")).digest()
    elif isinstance(benchmark, BenchmarkRun):
        hash_result = hashlib.sha256(benchmark.model_dump_json().encode("utf-8")).digest()
    else:
        raise TypeError("benchmark must be a string or BenchmarkRun")

    return base64.urlsafe_b64encode(hash_result).decode("utf-8")


def store_benchmark_run(run: BenchmarkRun, bmc: Optional[BenchmarkConfig] = None, file_name: Optional[str] = None):
    """
    Store the benchmark on file system.
    """
    json_string = run.model_dump_json()
    # This format of datetime string is needed to ensure the file name can be taken on.
    datetime_string = run.start.strftime("%Y%m%d-%H%M%S")

    if bmc is None:
        bmc = get_config()

    # Determine file name
    if file_name is None:
        if bmc.hash_suffix:
            json_hash = benchmark_run_string_hasher(run)
            file_name = f"pcbm_{datetime_string}_{json_hash}.json"
        else:
            file_name = f"pcbm_{datetime_string}.json"

    # Determine target folder
    if bmc.day_folders:
        tgt_dir = os.path.abspath(os.path.join(bmc.results_store, run.start.strftime("%Y-%m-%d")))
    else:
        tgt_dir = bmc.results_store

    # Create folder if not exists
    if not os.path.exists(tgt_dir):
        print(f"Creating Results Directory {tgt_dir}")
        os.makedirs(tgt_dir, exist_ok=True)

    # Write to folder
    print(f"Writing run to file: {file_name}")
    with open(os.path.join(tgt_dir, file_name), "w") as f:
        f.write(json_string)


def load_benchmark_run(path: str) -> BenchmarkRun:
    """
    Load a benchmark run from file.

    If indicated by the config, validates the hash of the file content against the hash in the file name
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")

    with open(path, "r") as f:
        content = f.read()

    config = get_config()

    # Return immediately, if we don't validate the hash
    if not config.validate_hash_on_read:
        return BenchmarkRun.model_validate_json(content)

    blocks = os.path.basename(path).split("_")

    if len(blocks) <= 2:
        print(f"Benchmark Run {path} does not contain a hash. Nothing to validate.")
        return BenchmarkRun.model_validate_json(content)

    hash_blocks = blocks[2:]
    hash_string = os.path.splitext("_".join(hash_blocks))[0]

    # Compute hash and verify
    if hash_string != benchmark_run_string_hasher(config):
        raise ValueError("File Contents were modified. File hash doesn't match.\n"
                         f"Expected: {hash_string}\n”"
                         f"Actual: {benchmark_run_string_hasher(config)}")

    return BenchmarkRun.model_validate_json(content)