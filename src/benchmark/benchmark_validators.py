from enum import Enum
from typing import Optional, Self, Any

import numpy as np
import tzlocal
from pydantic import BaseModel, Field, model_validator, ConfigDict

from benchmark.utils import DateTimeUTC


class BenchmarkType(str, Enum):
    SMOOTHER = "smoother"
    VX = "vx"
    VY = "vy"


class Stage(Enum):
    PREAMBLE = "preamble"
    BENCHMARK = "benchmark"
    EPILOG = "epilog"

# ======================================================================================================================
# Models which are arguments to the benchmark
# ======================================================================================================================

class BaseBenchmarkValidator(BaseModel):
    max_iter: int = Field(..., gt=0)
    nx: int = Field(..., gt=0)
    ny: int= Field(..., gt=0)

    dx: Optional[float] = Field(default=None, gt=0.0)
    dy: Optional[float] = Field(default=None, gt=0.0)
    eta_b: Optional[np.ndarray] = None
    eta_p: Optional[np.ndarray] = None

    # TODO check boundary condition and relax_v is correct
    boundary_condition: float = Field(default=-1.0)
    relax_v: float = Field(default=0.7)

    # Performance options
    cache_block_size_1: Optional[int] = Field(None, gt=0)
    cache_block_size_2: Optional[int] = Field(None, gt=0)

    # Inspection Options
    profile: bool = False
    samples: int = Field(..., gt=0)

    @model_validator(mode="after")
    def validate_sizes_eta(self) -> Self:
        self.check_shape("eta_b")
        self.check_shape("eta_p")
        return self

    def check_shape(self, attr: str):
        """
        Check shape only intended to run on `np.ndarrays`
        """
        attr_val = getattr(self, attr)
        assert isinstance(attr_val, np.ndarray) or attr_val is None, "Call for invalid attribute"

        if attr_val is not None and attr_val.shape != (self.ny + 1, self.nx + 1):
            raise ValueError(f"{attr} shape must be ({self.ny + 1}, {self.nx + 1})")


    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True
    )


class BenchmarkValidatorVX(BaseBenchmarkValidator):
    """
    Any combination of values can be passed. If a given matrix or tensor isn't passed, it is assumed to be zero
    """
    vx: Optional[np.ndarray] = None
    vx_new: Optional[np.ndarray] = None
    vy: Optional[np.ndarray] = None
    vx_rhs: Optional[np.ndarray] = None

    @model_validator(mode="after")
    def validate_velocities(self) -> Self:
        self.check_shape("vx")
        self.check_shape("vx_new")
        self.check_shape("vy")
        self.check_shape("vx_rhs")

        return self

class BenchmarkValidatorVY(BaseBenchmarkValidator):
    """
    Any combination of values can be passed. If a given matrix or tensor isn't passed, it is assumed to be zero
    """
    vy: Optional[np.ndarray] = None
    vy_new: Optional[np.ndarray] = None
    vx: Optional[np.ndarray] = None
    vy_rhs: Optional[np.ndarray] = None

    @model_validator(mode="after")
    def validate_velocities(self) -> Self:
        self.check_shape("vy")
        self.check_shape("vy_new")
        self.check_shape("vx")
        self.check_shape("vy_rhs")

        return self

class BenchmarkValidatorSmoother(BaseBenchmarkValidator):
    vx: Optional[np.ndarray] = None
    vx_new: Optional[np.ndarray] = None
    vx_rhs: Optional[np.ndarray] = None

    vy: Optional[np.ndarray] = None
    vy_new: Optional[np.ndarray] = None
    vy_rhs: Optional[np.ndarray] = None

    @model_validator(mode="after")
    def validate_velocities(self) -> Self:
        self.check_shape("vx")
        self.check_shape("vx_new")
        self.check_shape("vx_rhs")

        self.check_shape("vy")
        self.check_shape("vy_new")
        self.check_shape("vy_rhs")

        return self

# ======================================================================================================================
# Models which are the results of benchmarks
# ======================================================================================================================


class Timing(BaseModel):
    """
    Represents timing information of a single stage
    """
    name: str
    stage: Stage
    start: DateTimeUTC = Field(..., gt=0)
    end: DateTimeUTC = Field(..., gt=0)

    @property
    def duration(self) -> float:
        """
        Extract duration from timing
        """
        return (self.end - self.start).total_seconds()

    @property
    def local_start(self):
        """
        Convert start to user local timezone
        """
        local_tz = tzlocal.get_localzone_name()
        return self.start.astimezone(local_tz)

    @property
    def local_end(self):
        """
        Convert end to user local timezone
        """
        local_tz = tzlocal.get_localzone_name()
        return self.end.astimezone(local_tz)


class BenchmarkResults(BaseModel):
    """
    Represents the result of the execution of a single benchmark wrapper
    """
    timings: list[Timing] = Field(..., min_length=1, description="List of timing information")
    input_model: BenchmarkValidatorSmoother | BenchmarkValidatorVX | BenchmarkValidatorVY
    module: str = Field(..., description="Which is being benchmarked")
    benchmark_type: BenchmarkType = Field(..., description="Which benchmark of the module is executed")

    @model_validator(mode="after")
    def verify_timing_stages(self) -> Self:
        """
        Check that the preamble and epilogue stage only appear once
        """
        if len(list(filter(lambda t: t.stage == Stage.PREAMBLE,self.timings)))  > 1:
            raise ValueError(f"Preamble stage {self.benchmark_type.name} may appear at most once")

        if len(list(filter(lambda t: t.stage == Stage.EPILOG,self.timings)))  > 1:
            raise ValueError(f"Epilog stage {self.benchmark_type.name} may appear at most once")

        return self


class BenchmarkRun(BaseModel):
    """
    Represents the results of a call to the runner script.
    """
    start: DateTimeUTC = Field(..., description="Datetime Benchmark was started at")
    end: DateTimeUTC = Field(..., gt=0, description="Datetime Benchmark was ended at")

    args: dict[str, Any] = Field(..., description="Namespace dump of arguments with which the benchmark was called.")

    result: list[BenchmarkResults] = Field(..., min_length=1)

    env: Optional[dict[str, Any]] = Field(..., description="Environment dump of environment variables.")

    git_branch: str = Field(..., description="Git branch of the benchmark run")
    git_commit_msg: str = Field(..., description="Git commit message of current commit.")
    git_commit_hash: str = Field(..., description="Git commit hash of the benchmark run")

    dirty: bool = Field(False, description="Is set to true, if the runner was called with force and there were diffs in "
                                           "the working tree.")

    @property
    def smoother_benchmarks(self) -> list[BenchmarkResults]:
        """
        Get all smoother benchmarks
        """
        return list(filter(lambda b: b.benchmark_type == BenchmarkType.SMOOTHER, self.result))

    @property
    def vx_benchmarks(self) -> list[BenchmarkResults]:
        """
        Get all vx benchmarks
        """
        return list(filter(lambda b: b.benchmark_type == BenchmarkType.VX, self.result))

    @property
    def vy_benchmarks(self) -> list[BenchmarkResults]:
        """
        Get all vy benchmarks
        """
        return list(filter(lambda b: b.benchmark_type == BenchmarkType.VY, self.result))
