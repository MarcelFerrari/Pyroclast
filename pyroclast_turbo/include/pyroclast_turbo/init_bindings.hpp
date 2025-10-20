// Pyroclast: Scalable Geophysics Models
// https://github.com/MarcelFerrari/Pyroclast
//
// File: include/pyroclast_turbo/init_bindings.hpp
// Description: Forward declarations for Pyroclast Turbo binding initializers.
//
// Author: Marcel Ferrari
// Copyright (c) 2025 Marcel Ferrari.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef PYROCLAST_TURBO_INIT_BINDINGS_HPP
#define PYROCLAST_TURBO_INIT_BINDINGS_HPP
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pyroclast_turbo {

void init_mg_routines_bindings(nb::module_& m);
void init_interpolation_bindings(nb::module_& m);

} // namespace pyroclast_turbo

#endif // PYROCLAST_TURBO_INIT_BINDINGS_HPP
