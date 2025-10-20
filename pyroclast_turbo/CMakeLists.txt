# Pyroclast: Scalable Geophysics Models
# https://github.com/MarcelFerrari/Pyroclast
#
# File: pyroclast_turbo/CMakeLists.txt
# Description: CMake configuration for building the Pyroclast Turbo extensions.
#
# Author: Marcel Ferrari
# Copyright (c) 2025 Marcel Ferrari.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.15...3.27)
project(pyroclast_turbo LANGUAGES CXX)

# ----------------------------
# Options / defaults
# ----------------------------
option(PYROCLAST_USE_OPENMP "Enable OpenMP" ON)

# Install destination: <repo_root>/src/Pyroclast
if (NOT DEFINED PYROCLAST_PYDEST)
  set(PYROCLAST_PYDEST "${CMAKE_SOURCE_DIR}/src/Pyroclast")
endif()

# Default to Release
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------
# Python + nanobind
# ----------------------------
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(_PY_DEV_MODULE Development)
else()
  set(_PY_DEV_MODULE Development.Module)
endif()
find_package(Python 3.8 COMPONENTS Interpreter ${_PY_DEV_MODULE} REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

# ----------------------------
# Dependencies (optional)
# ----------------------------
if (PYROCLAST_USE_OPENMP)
  find_package(OpenMP QUIET)
endif()

# ----------------------------
# 1) High-performance kernels lib
# ----------------------------
file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cpp")

add_library(pyroclast_turbo_kernels STATIC ${KERNEL_SOURCES})
target_include_directories(pyroclast_turbo_kernels
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(pyroclast_turbo_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Linux optimizations + symbol visibility
target_compile_options(pyroclast_turbo_kernels PRIVATE -O3 -march=native)
add_compile_options(-fvisibility=hidden)

# OpenMP (if found)
if (PYROCLAST_USE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(pyroclast_turbo_kernels PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(pyroclast_turbo_kernels PUBLIC PYROCLAST_USE_OPENMP=1)
endif()

# LTO/IPO if supported
include(CheckIPOSupported)
check_ipo_supported(RESULT _ipo_supported OUTPUT _ipo_msg)
if (_ipo_supported)
  set_property(TARGET pyroclast_turbo_kernels PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ----------------------------
# 2) nanobind module (named "turbo" -> turbo*.so)
# ----------------------------
file(GLOB BINDING_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/*_bindings.cpp")

set(MODULE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/init_bindings.cpp
  ${BINDING_SOURCES}
)

nanobind_add_module(turbo ${MODULE_SOURCES})
target_include_directories(turbo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(turbo PRIVATE pyroclast_turbo_kernels)

# ----------------------------
# Install: drop turbo*.so into src/Pyroclast
# ----------------------------
install(TARGETS turbo
  LIBRARY DESTINATION "${PYROCLAST_PYDEST}"
  RUNTIME DESTINATION "${PYROCLAST_PYDEST}"
  ARCHIVE DESTINATION "${PYROCLAST_PYDEST}"
)

message(STATUS "Pyroclast turbo: install dest = '${PYROCLAST_PYDEST}'")
message(STATUS "Pyroclast turbo: OpenMP enabled? ${OpenMP_CXX_FOUND}")
