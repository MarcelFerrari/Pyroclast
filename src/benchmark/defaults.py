"""
File contains default values for the arg parser of the benchmark runner.
"""
from benchmark_validators import BenchmarkType

max_iter = 128
number_of_samples = 1
types = [BenchmarkType.SMOOTHER, BenchmarkType.VX, BenchmarkType.VY]
