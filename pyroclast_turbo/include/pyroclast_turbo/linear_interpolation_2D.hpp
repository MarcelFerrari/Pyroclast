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
