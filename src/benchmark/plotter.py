import os.path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from benchmark.benchmark_validators import PlotType


def plot_factory(pt: PlotType, data: pd.DataFrame, dst_folder: str, norm: bool = True) -> None:
    """
    Create a plot and store it in the given directory. Name is derived from the plot type.
    """
    plt.clf()

    plt.xticks(rotation=90)

    y = "normalized_duration" if norm else "duration"

    if pt == PlotType.MODULE_VS_CPU_X_DIM:
        plot = (sns.barplot(data, x="group-cpu-nx-ny", y=y, hue="module", errorbar="sd")
                .set_title("Runtime Module VS Cpu x Dim"))
    elif pt == PlotType.MODULE_VS_DIM_CPU16:
        plot = (sns.barplot(data.query("cpu_count == 16"), x="dim", y=y, hue="module", errorbar="sd")
                .set_title("Runtime Module VS Dim, CPU=16"))
    elif pt == PlotType.MODULE_VS_DIM_CPU8:
        plot = (sns.barplot(data.query("cpu_count == 8"), x="dim", y=y, hue="module", errorbar="sd")
                .set_title("Runtime Module VS Dim, CPU=8"))
    elif pt == PlotType.CPU_VS_MODULE_X_DIM:
        plot = (sns.barplot(data, x="group-module-nx-ny", y=y, hue="cpu_count", errorbar="sd")
                .set_title("Runtime CPU Count VS Module x Dim"))
    elif pt == PlotType.MODULE_X_CPU_VS_DIM:
        plot = (sns.barplot(data, x="dim", y=y, hue="group-module-cpu", errorbar="sd")
        .set_title("Runtime Module x CPU vs Dim"))
    elif pt == PlotType.DIM_VS_MODULE_X_CPU:
        plot = (sns.barplot(data, x="group-module-cpu", y=y, hue="dim", errorbar="sd")
                .set_title("Runtime Dim VS Module x CPU "))
    else:
        raise ValueError(f"Unknown plot type: {pt}")

    fig = plot.get_figure()
    dst = os.path.join(dst_folder, f"{pt.value}.svg")
    fig.savefig(dst, bbox_inches = "tight")
