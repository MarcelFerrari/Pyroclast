# Attempt to import CuPy and Numba CUDA, set to None if not available
try:
    import cupy as cp
    from numba import cuda
except ImportError:
    cp = None
    cuda = None

def get_numba_stream():
    """Get the current CuPy stream for interoperability with Numba CUDA."""
    if cp is None or cuda is None:
        raise RuntimeError("Attempted to get CuPy stream, but CuPy or Numba CUDA is not available.")
    cp_stream = cp.cuda.get_current_stream()
    ptr = int(cp_stream.ptr)
    return cuda.external_stream(ptr)


# Utils for GPU computing
def launch_2D(shape, block=(8, 32)):
    """
    Utility to determine grid/block sizes for 2D kernels
    We assume our input shape is (ny, nx), i.e. (rows, cols)
    and that the Cuda convention of (x, y) = (cols, rows) is used.
    """
    ny, nx = shape
    by, bx = block
    gx = (nx + bx - 1) // bx
    gy = (ny + by - 1) // by
    return (gx, gy), (bx, by)