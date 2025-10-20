#include <cmath>
#include <algorithm>
#include <omp.h>
#include "pyroclast_turbo/utils.hpp"
#include "pyroclast_turbo/linear_interpolation_2D.hpp"


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
) {

        const double dx = x[1] - x[0];
        const double dy = y[1] - y[0];

        // Zero accumulators (parallel)
        const int nGrid = nx1 * ny1;
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < nGrid; ++k) {
            grid_values[k] = 0.0;
            grid_weights[k] = 0.0;
        }

        // Helper for linear indexing
        #define IDX(i,j) ((i) * nx1 + (j))

        // Parallel over markers; atomics for concurrent scatters
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < n_markers; ++m) {
            const double mx = xm[m];
            const double my = ym[m];

            // Reference node indices for this marker
            const int mj = xidx[m];
            const int mi = yidx[m];

            // Distances relative to reference node (uniform grid)
            const double rx = std::abs(mx - x[mj]) / dx; // in [0, 1] ideally
            const double ry = std::abs(my - y[mi]) / dy;

            // Bilinear weights
            const double w00 = (1.0 - rx) * (1.0 - ry);
            const double w10 = (1.0 - rx) * ry;
            const double w01 = rx * (1.0 - ry);
            const double w11 = rx * ry;

            const double v = vals[m];

            const int p00 = IDX(mi,     mj);
            const int p10 = IDX(mi + 1, mj);
            const int p01 = IDX(mi,     mj + 1);
            const int p11 = IDX(mi + 1, mj + 1);

            // Accumulate weighted values
            #pragma omp atomic update
            grid_values[p00] += w00 * v;
            #pragma omp atomic update
            grid_values[p10] += w10 * v;
            #pragma omp atomic update
            grid_values[p01] += w01 * v;
            #pragma omp atomic update
            grid_values[p11] += w11 * v;

            // Accumulate weights
            #pragma omp atomic update
            grid_weights[p00] += w00;
            #pragma omp atomic update
            grid_weights[p10] += w10;
            #pragma omp atomic update
            grid_weights[p01] += w01;
            #pragma omp atomic update
            grid_weights[p11] += w11;
        }

        #undef IDX
    }
}