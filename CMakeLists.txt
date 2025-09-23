cmake_minimum_required(VERSION 3.20)
project(PyroclastTurbo LANGUAGES CXX)

# Option to skip native components entirely
option(PYROCLAST_TURBO "Build C++ turbo extensions" ON)

# Where to drop the built module on install (default: your package dir)
set(PYROCLAST_PYDEST
    "${CMAKE_SOURCE_DIR}/src/Pyroclast"
    CACHE PATH "Install path for the Python extension inside the package")

if(PYROCLAST_TURBO)
  add_subdirectory(pyroclast_turbo)
endif()
