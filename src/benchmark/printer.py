import argparse
import benchmark.results_processing as string_util
from Pyroclast.string_util import print_banner
from benchmark.benchmark_validators import BenchmarkRun, BenchmarkResults, BenchmarkType


"""
File contains a script to fetch a given benchmark result. It then reproduces the output of the runner script if provided
with the -v option.
"""


parser = argparse.ArgumentParser()