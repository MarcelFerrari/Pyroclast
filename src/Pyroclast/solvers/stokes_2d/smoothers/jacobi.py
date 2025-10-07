import numba as nb
import numpy as np
from .bc import apply_p_BC, apply_vx_BC, apply_vy_BC
from .coeff import x_momentum_coefficients, y_momentum_coefficients

@nb.njit(cache=True, parallel=True, fastmath=True)
def pressure_sweep(nx1, ny1, dx, dy,
                   vx, vy, p,
                   beta,
                   relax_p, rhs):
    
    # 1) Update only interior cells
    for i in nb.prange(1, ny1 - 1):
        for j in nb.prange(1, nx1 - 1):
            # The continuity residual
            res = rhs[i, j] - ((vx[i, j] - vx[i, j - 1])/dx +
                               (vy[i, j] - vy[i - 1, j])/dy)

            # Point-wise update of pressure
            p[i, j] += res * beta[i, j] * relax_p

    # Ensure zero-mean
    pbar = np.mean(p[1:-1, 1:-1])
    p -= pbar

    # Apply pressure boundary conditions
    apply_p_BC(p)

    return p

@nb.njit(cache=True, parallel=True, fastmath=True)
def _vx_jacobi_sweep(nx1, ny1,
                     dx, dy,
                     etap, etab,
                     vx_old, vy_old,
                     relax_v, rhs,
                     vx_new_out):
    """
    One Jacobi sweep for vx.
    Reads from (vx_old, vy_old), writes into vx_new_out.
    Interior ranges mirror your GS stencil: i in [1..ny1-2], j in [1..nx1-3].
    """
    for i in nb.prange(1, ny1 - 1):        # i: 1 .. ny1-2
        for j in range(1, nx1 - 2):        # j: 1 .. nx1-3
            etaA = etap[i,   j]
            etaB = etap[i,   j+1]
            eta1 = etab[i-1, j]
            eta2 = etab[i,   j]

            vx1, vx2, vx3, vx4, vx5, vy1, vy2, vy3, vy4 = x_momentum_coefficients(
                dx, dy, etaA, etaB, eta1, eta2
            )

            sum_neighbors = (
                vx1 * vx_old[i,   j-1] +
                vx2 * vx_old[i-1, j  ] +
                vx4 * vx_old[i+1, j  ] +
                vx5 * vx_old[i,   j+1]
                +
                vy1 * vy_old[i-1, j  ] +
                vy2 * vy_old[i,   j  ] +
                vy3 * vy_old[i-1, j+1] +
                vy4 * vy_old[i,   j+1]
            )

            vx_new_out[i, j] = (1.0 - relax_v) * vx_old[i, j] + relax_v * (rhs[i, j] - sum_neighbors) / vx3

    return vx_new_out


@nb.njit(cache=True, parallel=True, fastmath=True)
def _vy_jacobi_sweep(nx1, ny1,
                     dx, dy,
                     etap, etab,
                     vx_old, vy_old,
                     relax_v, rhs,
                     vy_new_out):
    """
    One Jacobi sweep for vy.
    Reads from (vx_old, vy_old), writes into vy_new_out.
    Interior ranges mirror your GS stencil: i in [1..ny1-3], j in [1..nx1-2].
    """
    for i in nb.prange(1, ny1 - 2):        # i: 1 .. ny1-3
        for j in range(1, nx1 - 1):        # j: 1 .. nx1-2
            etaA = etap[i,   j]
            etaB = etap[i+1, j]
            eta1 = etab[i,   j-1]
            eta2 = etab[i,   j]

            vy1, vy2, vy3, vy4, vy5, vx1, vx2, vx3, vx4 = y_momentum_coefficients(
                dx, dy, etaA, etaB, eta1, eta2
            )

            sum_neighbors = (
                vy1 * vy_old[i,   j-1] +
                vy2 * vy_old[i-1, j  ] +
                vy4 * vy_old[i+1, j  ] +
                vy5 * vy_old[i,   j+1]
                +
                vx1 * vx_old[i,   j-1] +
                vx2 * vx_old[i+1, j-1] +
                vx3 * vx_old[i,   j  ] +
                vx4 * vx_old[i+1, j  ]
            )

            vy_new_out[i, j] = (1.0 - relax_v) * vy_old[i, j] + relax_v * (rhs[i, j] - sum_neighbors) / vy3

    return vy_new_out


@nb.njit(cache=True, fastmath=True)
def jacobi_velocity_smoother(nx1, ny1,
                             dx, dy,
                             etap, etab,
                             vx, vy,              # initial guesses (also serve as buffers)
                             vx_new, vy_new,      # alternate preallocated buffers
                             relax_v, BC,
                             vx_rhs, vy_rhs,
                             max_iter):
    """
    Jacobi two-buffer smoother for (vx, vy).
    - No allocations inside; vx_new, vy_new are required preallocated buffers (same shape as vx, vy).
    - On each iteration, compute new vx, vy from old (vx, vy), apply BCs, then swap buffers.
    - Ensures an even number of iterations so the latest solution ends up back in (vx, vy).
    - Returns references to arrays holding the latest solution (vx, vy).
    """

    # Ensure even iterations for full sweeps so results land in (vx, vy)
    max_iter += max_iter % 2

    for _ in range(max_iter):
        # vx sweep: read (vx_old_ref, vy_old_ref) -> write vx_new_ref
        _vx_jacobi_sweep(nx1, ny1, dx, dy, etap, etab,
                         vx, vy, relax_v, vx_rhs, vx_new)
        apply_vx_BC(vx_new, BC)

        # vy sweep: read (vx_old_ref, vy_old_ref) -> write vy_new_ref
        _vy_jacobi_sweep(nx1, ny1, dx, dy, etap, etab,
                         vx, vy, relax_v, vy_rhs, vy_new)
        apply_vy_BC(vy_new, BC)

        # ping-pong (reference swap; no copies)
        vx, vx_new = vx_new, vx
        vy, vy_new = vy_new, vy

    # With even iterations, latest values are in the original (vx, vy) buffers.
    # Return the references holding the latest solution.
    return vx, vy