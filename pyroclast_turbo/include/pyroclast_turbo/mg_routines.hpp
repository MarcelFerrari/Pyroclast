// Pyroclast: Scalable Geophysics Models
// https://github.com/MarcelFerrari/Pyroclast
//
// File: include/pyroclast_turbo/mg_routines.hpp
// Description: Declarations for multigrid transfer kernels.
//
// Author: Marcel Ferrari
// Copyright (c) 2025 Marcel Ferrari.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef PYROCLAST_TURBO_MG_ROUTINES_HPP
#define PYROCLAST_TURBO_MG_ROUTINES_HPP

namespace pyroclast_turbo {

// Thread-safe restriction with atomic accumulations.
void restrict_2D(
    int nxh, int nyh,
    const double* xh, const double* yh, const double* uh,
    int nxH, int nyH,
    const double* xH, const double* yH,
    double* uH_num,   // output field
    double* uH_den    // weight accumulator
);

void prolong_2D(
    int nxH, int nyH,
    const double* xH, const double* yH,          // coarse coords
    const double* uH,                            // coarse field
    int nxh, int nyh,
    const double* xh, const double* yh,          // fine coords
    double* uh                                   // output fine field
);

} // namespace pyroclast_turbo

#endif // PYROCLAST_TURBO_MG_ROUTINES_HPP
