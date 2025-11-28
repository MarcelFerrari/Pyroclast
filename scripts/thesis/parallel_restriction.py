"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: parllel_restriction.py
Description: File contains the stuff needed to run a parallel restriction benchmark.


Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from Pyroclast.string_util import print_banner
from benchmark.utils import dtf

"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: grid_hierarchy.py
Description: This file implements the grid hierarchy for the multigrid method.

Author: Marcel Ferrari, Alexander Sotoudeh
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
import argparse
import os
import numpy as np
from typing import Any
import json

import numba as nb
from benchmark.git_checks import check_git_status, get_git_info


from Pyroclast.context import ContextNamespace, Context
from Pyroclast.solvers.stokes_2d.grid import Grid


class GridHierarchy:
    """
    Builds and stores grid levels from fine to coarse.
    """
    restrict_duration: float = 0

    def __init__(self, ctx: Context, nlevels: int, scaling: float):
        state: ContextNamespace
        params: ContextNamespace
        _opts: ContextNamespace

        state, params, _opts = ctx

        # Init grid
        x, y = state.nx1 - 1, state.ny1 - 1
        base = Grid(x, y, 0, ctx, False)


        base.rho[:, :] = state.rho
        base.etab[:, :] = state.etab
        base.etap[:, :] = state.etap

        # Initialize coarse levels
        self.nlevels = nlevels
        self.levels = [base]


        # Build coarse grids and propagate properties
        for lvl in range(1, self.nlevels):
            # Half the number of cells
            prev = self.levels[-1]
            nx_coarse = int((prev.nx - 1) / scaling) + 1
            ny_coarse = int((prev.ny - 1) / scaling) + 1

            # Initialize the Grid
            coarse = Grid(ny_coarse, nx_coarse, lvl, ctx, False)

            # Measure restriction performance.
            start = dtf()
            coarse.restrict_properties(prev)
            end = dtf()

            self.restrict_duration += (end - start).total_seconds()

            self.levels.append(coarse)


parser = argparse.ArgumentParser(prog="Benchmark Restriction",
                                 description="Run a benchmark on the [parallel] restriction method.")

parser.add_argument("-x", "--x-size",
                    help="Size of domain in horizontal meters, default 100'000",
                    type=float,
                    default=100000)
parser.add_argument("-y", "--y-size",
                    help="Size of domain in vertical meters, default 100'000",
                    type=float,
                    default=100000)
parser.add_argument("-s", "--scale",
                    help="Downscaling factor for multigrid. default 2.5",
                    default=2.5,
                    type=float)
parser.add_argument("-l", "--levels",
                    help="Number of levels in multigrid. default 5",
                    default=5,
                    type=int)
parser.add_argument("-X", "--x-res",
                    help="Size of domain in horizontal meters, default 100'000",
                    type=float,
                    default=1024)
parser.add_argument("-Y", "--y-res",
                    help="Size of domain in vertical meters, default 100'000",
                    type=float,
                    default=1024)
parser.add_argument("-c", "--cpu",
                    type=int,
                    nargs="+",
                    default=None,
                    help=f"Number of CPU cores to use. Default: {os.cpu_count()}")
parser.add_argument("-S", "--samples",
                    help="Number of samples to run for a given number of cpus",
                    type=int,
                    default=15)
parser.add_argument("-f", "--force",
                    action="store_true",
                    help="Force execution of benchmark with pending changes.")
parser.add_argument("-o", "--output",
                    type=str,
                    required=True,
                    help="Directory to write results to.")

def context_factory(args: dict[str, Any]) -> Context:
    parameters = ContextNamespace({
        "BC": -1,
        "xsize": args["x_size"],
        "ysize": args["y_size"],
    })
    rng = np.random.default_rng()
    state = ContextNamespace(
        {"nx": args["x_res"],
         "nx1": args["x_res"] + 1,
         "ny": args["y_res"],
         "ny1": args["y_res"] + 1,
         "rho": rng.random((args["y_res"]+1, args["x_res"]+1)),
         "etab": rng.random((args["y_res"]+1, args["x_res"]+1)),
         "etap": rng.random((args["y_res"]+1, args["x_res"]+1))
         })
    opts = ContextNamespace()

    return Context(state=state, params=parameters, options=opts)

def run_benchmark(arg_dict: dict[str, Any]):
    assert set(arg_dict.keys()) == {
        "x_size",
        "y_size",
        "scale",
        "levels",
        "x_res",
        "y_res",
        "cpu",
        "samples",
        "force",
        "output"
    }, "Dict Keys didn't match expectation"

    # Check if we have uncommitted changes:
    stage, unstaged = check_git_status()
    if (stage or unstaged) and not arg_dict["force"]:
        raise ValueError("You have uncommitted changes in this stage. Force with -f or commit changes")

    # Check the size still is reasonnable
    if arg_dict["x_size"] * (arg_dict["scale"] ** arg_dict["scale"]) < 16.0:
        raise ValueError("Smallest grid x resolution is no longer numerically relevant")
    if arg_dict["y_size"] * (arg_dict["scale"] ** arg_dict["scale"]) < 16.0:
        raise ValueError("Smallest grid y resolution is no longer numerically relevant")

    # Defaulting CPU count if not present
    if arg_dict["cpu"] is None:
        arg_dict["cpu"] = [os.cpu_count()]

    bname, chash, msg = get_git_info()
    results = []
    print(f"First run for compilation")
    GridHierarchy(
        ctx=context_factory(arg_dict),
        nlevels=arg_dict["levels"],
        scaling=arg_dict["scale"],
    )

    for i in arg_dict["cpu"]:
        nb.set_num_threads(i)
        print(f"Running Restriction with {i} cores")

        for s in range(arg_dict["samples"]):
            print(f"Running Sample {s}")
            res = GridHierarchy(
                ctx=context_factory(arg_dict),
                nlevels=arg_dict["levels"],
                scaling=arg_dict["scale"],
            )

            results.append({"cpu": i, "duration": res.restrict_duration})

    document = {
        "arguments": arg_dict,
        "branch_name": bname,
        "commit hash": chash,
        "commit message": msg,
        "results": results,
        "environment": dict(os.environ),
        "dirty": unstaged or stage,
    }

    date = dtf().strftime("%Y%m%d-%H%M%S")

    with open(os.path.join(arg_dict["output"], f"restriction_{date}.json"), "w") as f:
        json.dump(document, f)



if __name__ == "__main__":
    print_banner()

    ns = parser.parse_args()

    arg_dict = vars(ns)

