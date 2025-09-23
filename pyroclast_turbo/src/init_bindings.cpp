// init_bindings.cpp
#include <nanobind/nanobind.h>
#include "pyroclast_turbo/init_bindings.hpp"

namespace nb = nanobind;
using nb::module_;

namespace pyroclast_turbo {

// Forward-declared in the header and defined in submodule binding TUs:
//   void init_mg_routines_bindings(nb::module_&);

} // namespace pyroclast_turbo

// Single entry point: build the extension and wire submodules.
NB_MODULE(turbo, m) {
    m.doc() = "Pyroclast Turbo – C++ accelerators (nanobind)";

    // Submodules
    auto mg = m.def_submodule("mg_routines", "Multigrid routines");
    pyroclast_turbo::init_mg_routines_bindings(mg);

    // Simple diagnostics are handy
    m.def("openmp_enabled", []() {
    #ifdef PYROCLAST_USE_OPENMP
        return true;
    #else
        return false;
    #endif
    }, "Return whether the extension was compiled with OpenMP support.");
}
