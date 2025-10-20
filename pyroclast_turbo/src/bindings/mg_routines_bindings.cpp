// Pyroclast: Scalable Geophysics Models
// https://github.com/MarcelFerrari/Pyroclast
//
// File: src/bindings/mg_routines_bindings.cpp
// Description: Nanobind wrappers exposing multigrid routines to Python.
//
// Author: Marcel Ferrari
// Copyright (c) 2025 Marcel Ferrari.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "pyroclast_turbo/mg_routines.hpp"

namespace nb = nanobind;

namespace pyroclast_turbo {

void init_mg_routines_bindings(nb::module_ &m) {
    m.def(
        "restrict_2D",
        [](
           const int nxh, const int nyh,
           const nb::ndarray<double, nb::c_contig, nb::device::cpu> &xh,
           const nb::ndarray<double, nb::c_contig, nb::device::cpu> &yh,
           const nb::ndarray<double, nb::c_contig, nb::device::cpu> &uh,
           const int nxH, const int nyH,
           const nb::ndarray<double, nb::c_contig, nb::device::cpu> &xH,
           const nb::ndarray<double, nb::c_contig, nb::device::cpu> &yH,
           nb::ndarray<double, nb::c_contig, nb::device::cpu> &uH_num,
           nb::ndarray<double, nb::c_contig, nb::device::cpu> &uH_den
        ) -> nb::ndarray<double, nb::c_contig, nb::device::cpu>& {
            restrict_2D(
                nxh, nyh,
                static_cast<const double*>(xh.data()),
                static_cast<const double*>(yh.data()),
                static_cast<const double*>(uh.data()),
                nxH, nyH,
                static_cast<const double*>(xH.data()),
                static_cast<const double*>(yH.data()),
                static_cast<double*>(uH_num.data()),
                static_cast<double*>(uH_den.data())
            );
            return uH_num; // return the same array back to Python
        },
        nb::arg("nxh"), nb::arg("nyh"),
        nb::arg("xh"), nb::arg("yh"), nb::arg("uh"),
        nb::arg("nxH"), nb::arg("nyH"),
        nb::arg("xH"), nb::arg("yH"),
        nb::arg("uH_num"), nb::arg("uH_den")
    )

     .def(
        "prolong_2D",
        [](
            const int nxH, const int nyH,
            const nb::ndarray<double, nb::c_contig, nb::device::cpu> &xH,
            const nb::ndarray<double, nb::c_contig, nb::device::cpu> &yH,
            const nb::ndarray<double, nb::c_contig, nb::device::cpu> &uH,
            const int nxh, const int nyh,
            const nb::ndarray<double, nb::c_contig, nb::device::cpu> &xh,
            const nb::ndarray<double, nb::c_contig, nb::device::cpu> &yh,
            nb::ndarray<double, nb::c_contig, nb::device::cpu> &uh
        ) -> nb::ndarray<double, nb::c_contig, nb::device::cpu>& {
            prolong_2D(
                nxH, nyH,
                static_cast<const double*>(xH.data()),
                static_cast<const double*>(yH.data()),
                static_cast<const double*>(uH.data()),
                nxh, nyh,
                static_cast<const double*>(xh.data()),
                static_cast<const double*>(yh.data()),
                static_cast<double*>(uh.data())
            );
            return uh; // return the fine field array back to Python
        },
        nb::arg("nxH"), nb::arg("nyH"),
        nb::arg("xH"), nb::arg("yH"), nb::arg("uH"),
        nb::arg("nxh"), nb::arg("nyh"),
        nb::arg("xh"), nb::arg("yh"),
        nb::arg("uh")
    );
}
} // namespace pyroclast_turbo
