# Pyroclast: Scalable Geophysics Models
# https://github.com/MarcelFerrari/Pyroclast
#
# File: CMakeLists.txt
# Description: Top-level CMake configuration orchestrating optional native builds.
#
# Author: Marcel Ferrari
# Copyright (c) 2025 Marcel Ferrari.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
