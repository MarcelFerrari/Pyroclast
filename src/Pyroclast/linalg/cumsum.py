import numpy as np
import numba as nb

@nb.njit(parallel=True, fastmath=True)
def cumsum(arr, out=None):
    """
    Parallel cumulative sum (inclusive prefix sum).

    Parameters
    ----------
    arr : 1D numpy array
        Input array (integer, boolean, or float).
    out : 1D numpy array
        Output array to store the cumulative sums.

    Returns
    -------
    out : 1D numpy array
        Output array with cumulative sums.
    """
    n = arr.size
    n_threads = nb.get_num_threads()
    if out is None:
        out = np.empty_like(arr)

    # --- phase 1: compute per-chunk partial cumsums ---
    chunk_size = (n + n_threads - 1) // n_threads
    partial_sums = np.zeros(n_threads, dtype=out.dtype)

    for t in nb.prange(n_threads):
        start = t * chunk_size
        end = min(n, start + chunk_size)
        if start >= n:
            continue
        s = 0
        for i in range(start, end):
            s += arr[i]
            out[i] = s
        partial_sums[t] = s

    # --- phase 2: compute prefix sums of partial totals (sequential, small) ---
    for t in range(1, n_threads):
        partial_sums[t] += partial_sums[t - 1]

    # --- phase 3: add offsets in parallel ---
    for t in nb.prange(1, n_threads):
        start = t * chunk_size
        end = min(n, start + chunk_size)
        if start >= n:
            continue
        offset = partial_sums[t - 1]
        for i in range(start, end):
            out[i] += offset

    return out
