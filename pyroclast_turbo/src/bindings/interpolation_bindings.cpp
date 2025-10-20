// interpolation_bindings.cpp
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "pyroclast_turbo/linear_interpolation_2D.hpp"

namespace nb = nanobind;

namespace pyroclast_turbo {

void init_interpolation_bindings(nb::module_ &m) {
    m.def(
        "reduce_marker_values_2D",
        [](
            // Grid
            const int nx1, const int ny1,
            const nb::ndarray<double, nb::c_contig, nb::device::cpu> &x,
            const nb::ndarray<double, nb::c_contig, nb::device::cpu> &y,
            // Markers
            const int n_markers,
            const nb::ndarray<double, nb::c_contig, nb::device::cpu> &xm,
            const nb::ndarray<double, nb::c_contig, nb::device::cpu> &ym,
            const nb::ndarray<int,    nb::c_contig, nb::device::cpu> &xidx,
            const nb::ndarray<int,    nb::c_contig, nb::device::cpu> &yidx,
            const nb::ndarray<double, nb::c_contig, nb::device::cpu> &vals,
            // Outputs
            nb::ndarray<double, nb::c_contig, nb::device::cpu> &grid_values,
            nb::ndarray<double, nb::c_contig, nb::device::cpu> &grid_weights
        ) -> nb::ndarray<double, nb::c_contig, nb::device::cpu>& {

            reduce_marker_values_2D(
                // Grid
                nx1, ny1,
                static_cast<const double*>(x.data()),
                static_cast<const double*>(y.data()),
                // Markers
                n_markers,
                static_cast<const double*>(xm.data()),
                static_cast<const double*>(ym.data()),
                static_cast<const int*>(xidx.data()),
                static_cast<const int*>(yidx.data()),
                static_cast<const double*>(vals.data()),
                // Outputs
                static_cast<double*>(grid_values.data()),
                static_cast<double*>(grid_weights.data())
            );

            // Return the same array back to Python (mirrors your other bindings)
            return grid_values;
        },
        nb::arg("nx1"), nb::arg("ny1"),
        nb::arg("x"), nb::arg("y"),
        nb::arg("n_markers"),
        nb::arg("xm"), nb::arg("ym"),
        nb::arg("xidx"), nb::arg("yidx"),
        nb::arg("vals"),
        nb::arg("grid_values"), nb::arg("grid_weights")
    );
}

} // namespace pyroclast_turbo
