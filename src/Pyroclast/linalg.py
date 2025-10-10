import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

def get_xp(device: str = "cpu"):
    if device == "cpu":
        return np
    elif device == "gpu":
        if cp is None:
            raise ImportError("Failed to import cupy. Please install cupy to use GPU functionality.")
        return cp
    else:
        raise ValueError(f"Unknown device: {device}")