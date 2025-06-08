import os
from pydantic import BaseModel, Field


class BenchmarkConfig(BaseModel):
    results_store: str = Field(...,"[absolute] path to where the benchmark results are stored.")
    day_folders: bool = Field(True, description="Whether the benchmark results grouped by day and stored in "
                                                "subfolders for each day.")
    hash_suffix: bool = Field(True, description="Whether to add the json hash to the file name")
    validate_hash_on_read: bool = Field(True, description="Whether to validate the hash against the file content "
                                                          "when reading in the file")


if os.environ.get("PYROCLAST_BENCHMARK_CONFIG") is not None:
    _config_path = os.path.abspath(os.environ.get("PYROCLAST_BENCHMARK_CONFIG"))
else:
    _config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                                                "benchmark_config.json"))

if not os.path.exists(_config_path):
    raise FileNotFoundError(f"Config file {_config_path} not found.")


with open(_config_path, "r") as f:
    config = BenchmarkConfig.model_validate_json(f.read())