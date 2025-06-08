"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: benchmark/printer.py
Description: File contains a script to fetch a given benchmark result. It then reproduces the output of the runner 
             script if provided with the -v option.


Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""     


import argparse
import benchmark.results_processing as string_util
from Pyroclast.string_util import print_banner
from benchmark.benchmark_validators import BenchmarkRun, BenchmarkResults, BenchmarkType


"""
File contains a script to fetch a given benchmark result. It then reproduces the output of the runner script if provided
with the -v option.
"""


parser = argparse.ArgumentParser()
