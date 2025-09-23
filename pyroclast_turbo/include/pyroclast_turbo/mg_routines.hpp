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
