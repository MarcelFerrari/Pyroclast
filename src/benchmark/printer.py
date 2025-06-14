import argparse

import benchmark.results_processing as res_proc
from Pyroclast.string_util import print_banner
from benchmark.config import get_config

"""
File contains a script to fetch a given benchmark result. It then reproduces the output of the runner script if provided
with the -v option.
"""


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--benchmark-result",
                    type=str,
                    required=True,
                    help="Benchmark result file (relative to results folder in config)")
parser.add_argument("-c", "--config",
                    type=str,
                    default=None,
                    help="Path to config file overrides default path and environment option")


def main():
    ns = parser.parse_args()
    config = get_config(ns.config)

    path = ns.benchmark_result

    bmr = res_proc.load_benchmark_run(path, config)
    res_proc.print_statistics(bmr)


if __name__ == "__main__":
    print_banner()
    main()