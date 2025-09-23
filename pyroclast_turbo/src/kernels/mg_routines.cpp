// mg_routines.cpp
#include <algorithm>
#include <omp.h>
#include "pyroclast_turbo/mg_routines.hpp"
#include "pyroclast_turbo/utils.hpp"

namespace pyroclast_turbo {

void restrict_2D(
    int nxh, int nyh,
    const double* xh, const double* yh, const double* uh,
    int nxH, int nyH,
    const double* xH, const double* yH,
    double* uH_num,   // output field
    double* uH_den    // weight accumulator
) {
    if (nxH < 2 || nyH < 2 || nxh < 2 || nyh < 2) {
        return; // degenerate
    }

    const double dxH = xH[1] - xH[0];
    const double dyH = yH[1] - yH[0];
    const double xH0 = xH[0];
    const double yH0 = yH[0];
    const int nCoarse = nxH * nyH;

    // Zero accumulators
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < nCoarse; ++k) {
        uH_num[k] = 0.0;
        uH_den[k] = 0.0;
    }

    // (iH, jH) -> linear index on coarse
    #define IDX(i,j) ((i) * nxH + (j))

    // Scatter fine -> coarse (skip last fine row/col)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nyh - 1; ++i) {
        for (int j = 0; j < nxh - 1; ++j) {
            const double yhi = yh[i];
            const double xhj = xh[j];
            const double val = uh[i * nxh + j];

            int iH = clip_int(static_cast<int>((yhi - yH0) / dyH), 0, nyH - 2);
            int jH = clip_int(static_cast<int>((xhj - xH0) / dxH), 0, nxH - 2);

            const double ry = (yhi - yH[iH]) / dyH;
            const double rx = (xhj - xH[jH]) / dxH;

            const double w00 = (1.0 - rx) * (1.0 - ry);
            const double w01 = rx * (1.0 - ry);
            const double w10 = (1.0 - rx) * ry;
            const double w11 = rx * ry;

            const int p00 = IDX(iH,     jH);
            const int p10 = IDX(iH + 1, jH);
            const int p01 = IDX(iH,     jH + 1);
            const int p11 = IDX(iH + 1, jH + 1);

            // Numerators
            #pragma omp atomic update
            uH_num[p00] += w00 * val;
            #pragma omp atomic update
            uH_num[p01] += w01 * val;
            #pragma omp atomic update
            uH_num[p10] += w10 * val;
            #pragma omp atomic update
            uH_num[p11] += w11 * val;

            // Denominators (weights)
            #pragma omp atomic update
            uH_den[p00] += w00;
            #pragma omp atomic update
            uH_den[p10] += w10;
            #pragma omp atomic update
            uH_den[p01] += w01;
            #pragma omp atomic update
            uH_den[p11] += w11;
        }
    }

    // Normalize in place (nan_to_num -> 0 for empty bins)
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < nCoarse; ++k) {
        const double w = uH_den[k];
        uH_num[k] = (w > 0.0) ? (uH_num[k] / w) : 0.0;
    }

    #undef IDX
}


void prolong_2D(
    int nxH, int nyH,
    const double* xH, const double* yH,          // coarse coords
    const double* uH,                            // coarse field
    int nxh, int nyh,
    const double* xh, const double* yh,          // fine coords
    double* uh                                   // output fine field
) {
    if (nxH < 2 || nyH < 2 || nxh < 1 || nyh < 1) {
        return; // degenerate
    }

    const double dxH = xH[1] - xH[0];
    const double dyH = yH[1] - yH[0];
    const double xH0 = xH[0];
    const double yH0 = yH[0];

    // Linear index helpers
    #define IDX_FINE(i,j)   ((i) * nxh + (j))
    #define IDX_COARSE(i,j) ((i) * nxH + (j))

    // Interpolate each fine node from its 4 coarse neighbors
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nyh; ++i) {
        for (int j = 0; j < nxh; ++j) {
            const double yhi = yh[i];
            const double xhj = xh[j];

            // Lower-left coarse cell that contains (xhj, yhi)
            int iH = clip_int(static_cast<int>((yhi - yH0) / dyH), 0, nyH - 2);
            int jH = clip_int(static_cast<int>((xhj - xH0) / dxH), 0, nxH - 2);

            // Local coordinates in the coarse cell
            const double ry = (yhi - yH[iH]) / dyH;
            const double rx = (xhj - xH[jH]) / dxH;

            // Bilinear weights
            const double w00 = (1.0 - rx) * (1.0 - ry);
            const double w10 = (1.0 - rx) * ry;
            const double w01 = rx * (1.0 - ry);
            const double w11 = rx * ry;

            const int p00 = IDX_COARSE(iH,     jH);
            const int p10 = IDX_COARSE(iH + 1, jH);     // (iH+1, jH)
            const int p01 = IDX_COARSE(iH,     jH + 1); // (iH, jH+1)
            const int p11 = IDX_COARSE(iH + 1, jH + 1);

            const double v =
            w00 * uH[p00] +   // (1-rx)(1-ry)
            w10 * uH[p10] +   // rx(1-ry)     -> (iH,   jH+1)
            w01 * uH[p01] +   // (1-rx)ry     -> (iH+1, jH)
            w11 * uH[p11];    // rx*ry        -> (iH+1, jH+1)

            uh[IDX_FINE(i, j)] = v;
        }
    }
    #undef IDX_FINE
    #undef IDX_COARSE
}

} // namespace pyroclast_turbo
