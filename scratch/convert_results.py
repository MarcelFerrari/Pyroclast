import os
from benchmark.benchmark_validators import BenchmarkRun, BenchmarkResults, Timing
from benchmark.results_processing import store_benchmark_run
from benchmark.config import BenchmarkConfig
from typing import Optional


class OldTiming(Timing):
    error: Optional[str] = None
    traceback: Optional[str] = None


class OldBenchmarkResult(BenchmarkResults):
    timings: list[OldTiming]
    is_gpu: bool = False


class OldBenchmarkRun(BenchmarkRun):
    result: list[OldBenchmarkResult]


def convert_model(ob: OldBenchmarkRun):
    ob_insert_dict = ob.model_dump(exclude=["result"])

    new_results = []

    for result in ob.result:
        timings = []

        for timing in result.timings:
            new_timing = timing.model_dump()
            new_timing["error"] = None
            new_timing["traceback"] = None

            timings.append(new_timing)

        # Assemble new result dict
        new_result_dict = result.model_dump(exclude=["timings"])
        new_result_dict["timings"] = timings

        new_results.append(new_result_dict)

    ob_insert_dict["result"] = new_results
    return BenchmarkRun.model_validate(ob_insert_dict)


# TODO add directory with old benchmark storage format

source_dir = "/home/alisot2000/Polybox/S8/Thesis/benchmark-results/laptop/"
temp_config = BenchmarkConfig(
    results_store="/home/alisot2000/Polybox/S8/Thesis/benchmark-results/laptop-new/",
    plot_store="/home/alisot2000/Polybox/S8/Thesis/benchmark-results/plots/"
)

# Walk along the old results and convert them to the new standard
for path, dirs, files in os.walk(source_dir):
    for file in files:
        if "plot" in path:
            continue

        print(f"Processing: {os.path.join(path, file)}")
        with open(os.path.join(path, file), "r") as file:
            content = file.read()

        old_model = OldBenchmarkRun.model_validate_json(content)
        converted = convert_model(old_model)

        store_benchmark_run(run=converted, bmc=temp_config)