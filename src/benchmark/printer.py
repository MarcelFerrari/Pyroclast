import argparse
import os.path
from typing import Optional

import benchmark.defaults as defaults
import benchmark.results_processing as res_proc
from Pyroclast.string_util import print_banner
from benchmark.benchmark_validators import PlotType, BenchmarkPlot
from benchmark.config import get_config, BenchmarkConfig
from benchmark.utils import dtf
from benchmark.plotter import plot_factory
import benchmark.git_checks as gc


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
parser.add_argument(f"-p", "--plots",
                    action="store_true",
                    help="Generate plots. ")
parser.add_argument(f"-t", "--plot-type",
                    default=defaults.plot_types,
                    nargs="+",
                    type=PlotType,
                    help="List of Plot Types to generate. By default, all plots are generated")
parser.add_argument(f"-n", "--normalized",
                    action="store_true",
                    help="Normalize compute duration for a more comparable result across sizes")
parser.add_argument("-d", "--destination",
                    type=str,
                    help="Destination folder where results will be saved",
                    default=None)
parser.add_argument("-f", "--force",
                    action="store_true",
                    help="Force run of plotter even with pending changes.")


def prep_plot_dir(file_name: str,
                  cfg: Optional[BenchmarkConfig] = None,
                  out_folder: Optional[str] = None):
    """
    Create standard plots
    """
    now = dtf()

    # Get config
    if cfg is None:
        cfg = get_config()

    # Get Store directory
    if out_folder is None:
        stripped_filename = os.path.splitext(os.path.basename(file_name))[0]
        folder_name = "plot_" + now.strftime("%Y%m%d-%H%M%S") + "_" + stripped_filename
        base = cfg.plot_store
        out_folder = os.path.abspath(os.path.join(base, folder_name))

    # Create directory if not exists
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    return out_folder


def main():
    ns = parser.parse_args()
    config = get_config(ns.config)

    path = ns.benchmark_result

    bmr = res_proc.load_benchmark_run(path, config)
    dataframe = res_proc.augment_df(res_proc.create_dataframe(bmr))
    res_proc.print_statistics(bmr)

    print(f"Overall Runtime: {(bmr.end - bmr.start).total_seconds()}")

    # Abort, if nothing to plot
    if not ns.plots:
        return

    staged, unstaged = gc.check_git_status()
    if not ns.force:
        if staged:
            raise ValueError("Git tree isn't clean. some changes are staged. stash changes, commit changes or use -f")
        if unstaged:
            raise ValueError("Git tree isn't clean. some changes are unstaged. stash changes, commit changes or use -f")

    tgt = prep_plot_dir(file_name=path, out_folder=ns.destination, cfg=config)

    for plot_type in ns.plot_type:
        plot_factory(pt=plot_type,
                     dst_folder=tgt,
                     data=dataframe,
                     norm=ns.normalized)

    branch, chash, cmsg  = gc.get_git_info()
    bmp = BenchmarkPlot(git_branch=branch,
                        git_commit_msg=cmsg,
                        git_commit_hash=chash,

                        dirty=staged or unstaged,
                        normalized=ns.normalized,
                        filename=os.path.basename(path),
                        plot_types=ns.plot_type)

    with open(os.path.join(tgt, "plot_info.json"), "w") as f:
        f.write(bmp.model_dump_json(indent=2))


if __name__ == "__main__":
    print_banner()
    main()