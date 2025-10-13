import contextlib
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

@contextlib.contextmanager
def view(arr, *, intent: str, device: str):
    """
    Stage `arr` on `device` with Fortran-like INTENT semantics.

    intent:
      - "in"    : read-only on target (copy in if needed; no copy-back)
      - "out"   : write-only on target (no copy-in; copy-back on exit)
      - "inout" : read/write on target (copy in; copy-back on exit)

    device: "cpu" or "gpu"
    """
    if intent not in ("in", "out", "inout"):
        raise ValueError("intent must be 'in', 'out', or 'inout'")
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")

    # Determine target device
    target_device = device

    # Detect source device
    if cp is not None and isinstance(arr, cp.ndarray):
        source_device = "gpu"
    elif isinstance(arr, np.ndarray):
        source_device = "cpu"
    else:
        raise ValueError("Input array must be a NumPy or CuPy array.")

    # Fast path: already on requested device
    # No copies, just alias
    if source_device == target_device:
        yield arr
        return

    # Cross-device: CPU -> GPU
    if source_device == "cpu" and target_device == "gpu":
        if cp is None:
            raise RuntimeError("GPU target selected but CuPy not available.")
        
        if intent in ("in", "inout"):
            # allocate + copy host→device
            arr_dev = cp.asarray(arr)
        else:  # intent == "out"
            # allocate only; no copy-in
            arr_dev = cp.zeros_like(arr)
        
        # yield the device array
        yield arr_dev
        
        # copy back if needed
        if intent in ("out", "inout"):
            # copy device -> host into the original array
            arr_dev.get(out=arr)

    # Cross-device: GPU -> CPU
    elif source_device == "gpu" and target_device == "cpu":
        if cp is None:
            raise RuntimeError("GPU source detected but CuPy not available.")

        if intent in ("in", "inout"):
            # allocate + copy device→host
            arr_host = arr.get()
        else:  # intent == "out"
            # allocate only; no copy-in
            arr_host = np.zeros_like(arr)

        # yield the host array
        yield arr_host

        if intent in ("out", "inout"):
            # copy host -> device back into the original GPU array
            arr.set(arr_host)
    else:
        # Shouldn't reach here with only "cpu"/"gpu"
        raise RuntimeError(f"Unsupported transfer path: {source_device} to {target_device}")
