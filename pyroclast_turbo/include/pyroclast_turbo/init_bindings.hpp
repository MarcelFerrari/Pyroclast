#ifndef PYROCLAST_TURBO_INIT_BINDINGS_HPP
#define PYROCLAST_TURBO_INIT_BINDINGS_HPP
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pyroclast_turbo {

void init_mg_routines_bindings(nb::module_& m);
void init_interpolation_bindings(nb::module_& m);

} // namespace pyroclast_turbo

#endif // PYROCLAST_TURBO_INIT_BINDINGS_HPP
