from Pyroclast.gpu_utils import get_numba_stream, launch_2D
import cupy as cp
from numba import cuda

# -----------------------
# Device helpers
# -----------------------
@cuda.jit(inline="always")
def _clip_i(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

# -----------------------
# Kernels
# -----------------------

@cuda.jit
def _restrict2d_scatter(
    nxh, nyh, xh, yh, uh,
    nxH, nyH, xH, yH,
    uH_accum, wH_accum,
    dxH, dyH, xH0, yH0
):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # if i >= nyh - 1 or j >= nxh - 1:
    #     return
    if not (0 <= i < nyh - 1 and 0 <= j < nxh - 1):
        return

    yhi = yh[i]
    xhj = xh[j]

    iH = int((yhi - yH0) / dyH)
    jH = int((xhj - xH0) / dxH)
    iH = _clip_i(iH, 0, nyH - 2)
    jH = _clip_i(jH, 0, nxH - 2)

    ry = (yhi - yH[iH]) / dyH
    rx = (xhj - xH[jH]) / dxH

    w00 = (1.0 - rx) * (1.0 - ry)
    w01 = rx * (1.0 - ry)
    w10 = (1.0 - rx) * ry
    w11 = rx * ry

    v = uh[i, j]

    cuda.atomic.add(uH_accum, (iH,   jH  ), w00 * v)
    cuda.atomic.add(uH_accum, (iH+1, jH  ), w10 * v)
    cuda.atomic.add(uH_accum, (iH,   jH+1), w01 * v)
    cuda.atomic.add(uH_accum, (iH+1, jH+1), w11 * v)

    cuda.atomic.add(wH_accum, (iH,   jH  ), w00)
    cuda.atomic.add(wH_accum, (iH+1, jH  ), w10)
    cuda.atomic.add(wH_accum, (iH,   jH+1), w01)
    cuda.atomic.add(wH_accum, (iH+1, jH+1), w11)

@cuda.jit
def _prolong2d(
    nxH, nyH, xH, yH, uH,
    nxh, nyh, xh, yh, uh,
    dxH, dyH, xH0, yH0
):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i >= nyh or j >= nxh:
        return

    yhi = yh[i]
    xhj = xh[j]

    iH = int((yhi - yH0) / dyH)
    jH = int((xhj - xH0) / dxH)
    iH = _clip_i(iH, 0, nyH - 2)
    jH = _clip_i(jH, 0, nxH - 2)

    ry = (yhi - yH[iH]) / dyH
    rx = (xhj - xH[jH]) / dxH

    uh[i, j] = (1.0 - rx) * (1.0 - ry) * uH[iH,   jH  ] + \
               (      rx) * (1.0 - ry) * uH[iH,   jH+1] + \
               (1.0 - rx) * (      ry) * uH[iH+1, jH  ] + \
               (      rx) * (      ry) * uH[iH+1, jH+1]

# -----------------------
# Public API (same signatures)
# -----------------------

def restrict_2D(nxh, nyh, xh, yh, uh, nxH, nyH, xH, yH, uH, uHw):
    """
    In-place GPU restriction (fine->coarse) using atomics.
    All arrays are CuPy and preallocated. Returns uH (normalized).
    """
    assert isinstance(uH, cp.ndarray)
    assert isinstance(uHw, cp.ndarray)
    assert isinstance(xH, cp.ndarray)
    assert isinstance(yH, cp.ndarray)
    assert isinstance(xh, cp.ndarray)
    assert isinstance(yh, cp.ndarray)
    assert isinstance(uh, cp.ndarray)
    
    # zero accumulators (in-place, no allocation)
    uH.fill(0)
    uHw.fill(0)

    # scalars (host)
    dxH = float((xH[1] - xH[0]).item())
    dyH = float((yH[1] - yH[0]).item())
    xH0 = float(xH[0].item())
    yH0 = float(yH[0].item())

    # launch over fine interior (exclude last row/col)
    grid, block = launch_2D((nyh - 1, nxh - 1))
    stream = get_numba_stream()
    _restrict2d_scatter[grid, block, stream](
        nxh, nyh, xh, yh, uh,
        nxH, nyH, xH, yH,
        uH, uHw,
        dxH, dyH, xH0, yH0
    )

    # normalize uH by weights uHw
    uH /= uHw  # in-place device op
    return uH

def prolong_2D(nxH, nyH, xH, yH, uH,
               nxh, nyh, xh, yh, uh):
    """
    In-place GPU prolongation (coarse->fine).
    All arrays are CuPy and preallocated. Returns uh.
    """

    assert isinstance(uH, cp.ndarray)
    assert isinstance(uh, cp.ndarray)
    assert isinstance(xH, cp.ndarray)
    assert isinstance(yH, cp.ndarray)
    assert isinstance(xh, cp.ndarray)
    assert isinstance(yh, cp.ndarray)

    dxH = float((xH[1] - xH[0]).item())
    dyH = float((yH[1] - yH[0]).item())
    xH0 = float(xH[0].item())
    yH0 = float(yH[0].item())

    grid, block = launch_2D((nyh, nxh))
    stream = get_numba_stream()
    _prolong2d[grid, block, stream](
        nxH, nyH, xH, yH, uH,
        nxh, nyh, xh, yh, uh,
        dxH, dyH, xH0, yH0
    )
    return uh
