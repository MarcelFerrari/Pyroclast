"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: Pyroclast/model/stokes_2D_mg/smoothers/template.py
Description: File contains template for smoohter benchmark files.

Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""



from typing import Type, Optional


# INFO need to use string references to avoid circular imports and deal with benchmark packaged not available
def benchmark_factory() -> tuple[Optional[Type["BenchmarkSmoother"]],
                                 Optional[Type["BenchmarkVX"]],
                                 Optional[Type["BenchmarkVY"]]]:
    """
    Returns Benchmark Classes needed for benchmarking. Done via factory to avoid issues with the `benchmark` package
    not being available in a production environment.
    """
    return None, None, None
