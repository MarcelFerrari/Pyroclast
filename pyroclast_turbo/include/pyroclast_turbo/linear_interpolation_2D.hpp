// Pyroclast: Scalable Geophysics Models
// https://github.com/MarcelFerrari/Pyroclast
//
// File: include/pyroclast_turbo/linear_interpolation_2D.hpp
// Description: Declarations for marker-to-grid linear interpolation kernels.
//
// Author: Marcel Ferrari
// Copyright (c) 2025 Marcel Ferrari.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef PYROCLAST_TURBO_LINEAR_INTERPOLATION_2D
#define PYROCLAST_TURBO_LINEAR_INTERPOLATION_2D

namespace pyroclast_turbo {

void reduce_marker_values_2D(
    // Grid
    const int nx1, const int ny1,
    const double* x, const double* y,
    // Markers
    const int n_markers,
    const double* xm, const double* ym,
    const int* xidx, const int* yidx,
    const double* vals,
    // Outputs
    double* grid_values, double* grid_weights
);

} // namespace pyroclast_turbo

#endif // PYROCLAST_TURBO_LINEAR_INTERPOLATION_2D
