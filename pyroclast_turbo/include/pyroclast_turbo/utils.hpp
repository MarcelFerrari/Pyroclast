// Pyroclast: Scalable Geophysics Models
// https://github.com/MarcelFerrari/Pyroclast
//
// File: include/pyroclast_turbo/utils.hpp
// Description: Utility helpers shared across Pyroclast Turbo kernels.
//
// Author: Marcel Ferrari
// Copyright (c) 2025 Marcel Ferrari.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef PYROCLAST_TURBO_UTILS_HPP
#define PYROCLAST_TURBO_UTILS_HPP

namespace pyroclast_turbo {

// Clamp integer to [lo, hi]
static inline int clip_int(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi ? hi : v);
}

} // namespace pyroclast_turbo

#endif // PYROCLAST_TURBO_UTILS_HPP
