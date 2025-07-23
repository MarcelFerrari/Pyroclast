import numpy as np
from collections import deque

class ResidualTracker:
    def __init__(self, m, tol_p, tol_vx, tol_vy, plateau_thresh):
        """
        Args:
            m: Number of recent residuals to track.
            tol_p, tol_vx, tol_vy: Absolute convergence tolerances.
            plateau_thresh: Relative variation ((max - min) / min) threshold to detect plateau-style convergence.
        """
        self.m = m
        self.tol_p = tol_p
        self.tol_vx = tol_vx
        self.tol_vy = tol_vy
        self.plateau_thresh = plateau_thresh

        self.p_history = deque(maxlen=m)
        self.vx_history = deque(maxlen=m)
        self.vy_history = deque(maxlen=m)

    def _is_plateau(self, history):
        if len(history) < self.m:
            return False
        arr = np.array(history)
        max_val = np.max(arr)
        min_val = np.min(arr)
        rel_var = (max_val - min_val) / (min_val + 1e-12)
        return rel_var < self.plateau_thresh

    def _has_converged(self, history, tol):
        return history[-1] < tol or self._is_plateau(history)

    def update(self, p_res: float, vx_res: float, vy_res: float):
        self.p_history.append(p_res)
        self.vx_history.append(vx_res)
        self.vy_history.append(vy_res)

    def converged(self) -> bool:
        if len(self.p_history) < self.m:
            return False
        return (
            self._has_converged(self.p_history, self.tol_p) and
            self._has_converged(self.vx_history, self.tol_vx) and
            self._has_converged(self.vy_history, self.tol_vy)
        )

    def reset(self):
        self.p_history.clear()
        self.vx_history.clear()
        self.vy_history.clear()
