import numba as nb
import numpy as np

@nb.njit(cache=True, parallel=True)
def clip_float(arr, min_val, max_val, eps = 1e-8):
    """
    Clips the values in a float array to be within [min_val, max_val).
    
    Parameters:
    arr (np.array): Input array of floats.
    min_val (float): Minimum allowed value.
    max_val (float): Maximum allowed value.
    
    Returns:
    np.array: Clipped array.
    """
    n = arr.shape[0]
    for i in nb.prange(n):
        if arr[i] < min_val:
            arr[i] = min_val + eps
        elif arr[i] >= max_val:
            arr[i] = max_val - eps
    return arr