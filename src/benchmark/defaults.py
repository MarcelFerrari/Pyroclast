"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: benchmark/defaults.py
Description: File contains default values for the arg parser of the benchmark runner.

Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
from .benchmark_validators import BenchmarkType

max_iter = 128
number_of_samples = 1
types = [BenchmarkType.SMOOTHER, BenchmarkType.VX, BenchmarkType.VY]
