"""
File contains default values for the arg parser of the benchmark runner.
"""
from .benchmark_validators import BenchmarkType, PlotType

max_iter = 128
number_of_samples = 1
benchmark_types = [BenchmarkType.SMOOTHER, BenchmarkType.VX, BenchmarkType.VY]
plot_types = [PlotType.MODULE_VS_CPU_X_DIM,
              PlotType.MODULE_VS_DIM_CPU16,
              PlotType.MODULE_VS_DIM_CPU8,
              PlotType.CPU_VS_MODULE_X_DIM,
              PlotType.MODULE_X_CPU_VS_DIM,
              PlotType.DIM_VS_MODULE_X_CPU]