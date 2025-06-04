import datetime
import json
from enum import Enum
from typing import Optional, Self, Union, Literal

from annotated_types import Gt, Annotated
from pydantic import BaseModel, Field, AfterValidator, model_validator, computed_field, ConfigDict


class BoundaryCondition(str, Enum):
    """
    External Enum, needed for json
    """
    AIR = "air"
    STICK = "stick"
    # TODO I've forgotten your intro on boundary conditions.
    # check boundary condition like `if model.field == BoundaryCondition.AIR`


class BoundaryConditionInt(Enum):
    """
    Alternative Boundary Condition with ints.
    However, it is discouraged because of typing unions `Union[enum_a, enum_b]`, if both enum_a is defined over
    [0, ..., 5] and enum_b is defined over [0, ..., 3] pydantic doesn't know if it's a enum_a or enum_b for [0, ..., 3]
    that's why it is prefers to use strings, as name collisions are much less likely and it makes the input file easier
    to understand.
    """
    AIR = 0
    STICK = 1



# Generally, we can define arbitrary types for pydantic, if we give it the class, a validator, and a serializer.
# It is recommended to add a WithJsonSchema, because then it will show up better in documentation
# PydanticIntBoundaryCondition = Annotated[<Type>,
#                                          AfterValidator(lambda x : str(x),
#                                          PlainSerializer(lambda x: x.__name__, return_type=str),
#                                          WithJsonSchema({'type': 'str'}, mode='serialization')]

def localize_time(dt: datetime.datetime) -> datetime.datetime:
    """
    Convert any datetime object to UTC (Account for locales not set the same across nodes)
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)

    if dt.tzinfo.utcoffset(dt) != datetime.timezone.utc:
        return dt.astimezone(datetime.timezone.utc)

    return dt


# Define custom datetime that is always UTC
DateTimeUTC = Annotated[datetime.datetime, AfterValidator(localize_time)]


class State(BaseModel):
    # Start will be checked to be of datetime.datetime. After Pydantic is done with that check, my function gets called
    # which then ensures the timezone is set to UTC.

    # Possible use case, run metadata that was resumed from a checkpoint.
    start: DateTimeUTC = Field(default_factory=lambda : datetime.datetime.now(datetime.timezone.utc))


class ModelParameters(BaseModel):
    """
    Example for Validation Model for Physical Model Verification
    """
    is_3d: bool = Field(False, description="Authoritative field for 3D Model. If False, all *_z parameters and "
                                           "performance options will be ignored. If True, raises an error, if any "
                                           "parameter or performance option *_z is left None")

    dim_x_km: float = Field(..., description="Dimension of Model in x direction, unit km")
    dim_y_km: float = Field(..., description="Dimension of Model in y direction, unit km")
    dim_z_km: Optional[float] = Field(
        None, description="Dimension of Model in z direction, unit km. Can be left Noen for 2D Model")

    p_start_mp: float = Field(..., description="Presser Fixpoint, Unit Mpa")

    # Generally, pydantic is greedy when validating. If you have a union type, pydantic will take the first type that
    # validates correctly.
    boundary_condition: BoundaryCondition = BoundaryCondition.AIR
    # TODO More options to be done

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

# ======================================================================================================================
# Example for greedy validation
# ======================================================================================================================

class EnumA(Enum):
    U = 0
    V = 1
    W = 2


class EnumB(Enum):
    X = 1
    Y = 2
    Z = 3


class SomeModel(BaseModel):
    arg: EnumA | EnumB

print(SomeModel.model_validate({"arg": 0})) # Converted to EnumA (obviously)
print(SomeModel.model_validate({"arg": 1})) # Converted to EnumA (and could have been EnumB)
print(SomeModel.model_validate({"arg": 2})) # Converted to EnumA (and could have been EnumB)
print(SomeModel.model_validate({"arg": 3})) # Converted to EnumB (obviously)

# ======================================================================================================================

class RangeModel(BaseModel):
    """
    Define custom Ranges for your MPI ranks (useful for heterogeneous hardware
    """
    # Check that we have values >= 1
    start_x: int = Field(..., description="# TODO", ge=1)
    start_y: int = Field(..., description="# TODO", ge=1)
    start_z: Optional[int] = Field(None, description="# TODO", ge=1)

    size_x: int = Field(..., description="# TODO", ge=1)
    size_y: int = Field(..., description="# TODO", ge=1)
    size_z: Optional[int] = Field(None, description="# TODO", ge=1)


class PerformanceOptions(BaseModel):
    # Specification of a Positive int with Annotated Syntax
    nx: Annotated[int, Gt(0)]

    # Specification of a Positive int with Field Syntax
    ny: int = Field(..., description="Number of cells in X direction", gt=0)  # Personal preference, use Field, bc of description
    nz: Optional[int] = Field(None, description="Number of cells in X direction")

    proc_subdiv_x: Optional[int] = Field(None, description="Subdivision along X direction", ge=1)
    proc_subdiv_y: Optional[int] = Field(None, description="Subdivision along Y direction", ge=1)
    proc_subdiv_z: Optional[int] = Field(None, description="Subdivision along Z direction", ge=1)

    # We can nest models as submodels, list, dict, set (and I think other containers are also possible)
    task_range_map: Optional[list[RangeModel]] = Field(None, description="Task Range Map")

    rank: Optional[int] = Field(None, description="Used internally, once the code is run with mpi")

    # This is an alternative way to define a fixed subset of values over an enum.
    mpi_mode: Literal["proc_subdiv_mode", "task_mode"]

    @computed_field
    @property
    def process_range(self) -> RangeModel:
        """
        Return the tensor or matrix, this rank is responsible for
        """
        if self.task_range_map is None:
            pass # TODO implement partitioning with proc_subdiv

        return self.task_range_map[self.rank]


    @model_validator(mode="after")
    def validate_range(self) -> Self:
        """
        This function is run, once all attributes have been instantiated.
        """
        if self.task_range_map is None:
            return self

        for i, block in enumerate(self.task_range_map):
            if block.start_x > self.nx \
                    or block.start_y > self.ny \
                    or (block.start_z is not None and block.start_z > self.nz):
                raise IndexError(f"Block {i} has a start_index out of range. "
                                 f"${block.start_x, block.start_y, block.start_z}, {self.nx, self.ny, self.nz}")

            if block.start_x + block.size_x < self.nx \
                    or block.start_y + block.size_y < self.ny \
                    or (block.start_z is not None and block.start_z + block.size_z > self.nz):
                raise IndexError(
                    f"Block {i} has a start_index out of range. "
                    f"${block.start_x + block.size_x, block.start_y + block.size_y, block.start_z + block.size_z}, "
                    f"{self.nx, self.ny, self.nz}")

        return self

    @model_validator(mode="after")
    def ensure_splitting(self) -> Self:
        """
        Ensure some definition exists on how to split the work among the mpi ranks
        """
        if self.task_range_map is None and self.mpi_mode == "task_mode":
            raise ValueError("task_range_map cannot be None with mpi_mode 'task_mode'")

        if self.mpi_mode == "proc_subdiv_mode" and (self.proc_subdiv_x is None
                                                    or self.proc_subdiv_y is None
                                                    or self.proc_subdiv_z is None):
            raise ValueError("Not all proc_subdiv fields populated.")


# What the ensure_splitting validator checks can also be achieved with two subclasses
class PerformanceOptionsSubDiv(PerformanceOptions):
    proc_subdiv_x: int
    proc_subdiv_y: int

    mpi_mode: Literal["proc_subdiv_mode"] = "proc_subdiv_mode"


class PerformanceOptionsTask(PerformanceOptions):
    task_range_map: list[RangeModel]

    mpi_mode: Literal["task_mode"] = "task_mode"


class Config(BaseModel):
    state: State
    params: ModelParameters
    # Using this typing union, is a more expensive validation operation. However, since this validation only runs once,
    # we don't care about that. Counter example where the other method is favorable would be a webserver or database
    # where we constantly validate date
    options: Union[PerformanceOptionsTask, PerformanceOptionsSubDiv]


# You can validate a python [nested] dict like this
# Config.model_validate({...})

# You can validate a json string like this
# Config.model_validate_json("{}")

# You can convert a ModelInstance to a python dict
# Config.model_dump()

# You can convert a ModelInstance to a json string
# Config.model_dump_json()

# It's also possible to instantiate a Model like a regular class Config(state=..., params=..., options=...)
# model_dump*() and model_validate*() both have a big range of options to further tweek validation and serialization
# (like exclusions, custom serializers, aliases, ...)

# Sadly, the json schema isn't viewable directly with the openapi standard, it should be possible to wrap this returned
# json schema for it to be displayable with the swagger-ui
# https://petstore.swagger.io/ (It would look like the "Model" section at the bottom)
print(json.dumps(Config.model_json_schema(), indent=2))