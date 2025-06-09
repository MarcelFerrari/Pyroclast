import numpy as np
from collections import deque

class ResidualTracker:
    def __init__(self, m, tol_p, tol_vx, tol_vy, convergence_thresh, divergence_thresh):
        """
        Args:
            m: Number of recent residuals to track.
            tol_p: Convergence tolerance for pressure.
            tol_vx: Convergence tolerance for vx.
            tol_vy: Convergence tolerance for vy.
            convergence_thresh: Relative change threshold to detect convergence.
        """
        self.m = m
        self.tol_p = tol_p
        self.tol_vx = tol_vx
        self.tol_vy = tol_vy
        self.convergence_thresh = convergence_thresh
        self.divergence_thresh = divergence_thresh

        self.p_history = deque(maxlen=m)
        self.vx_history = deque(maxlen=m)
        self.vy_history = deque(maxlen=m)

    def update(self, p_res: float, vx_res: float, vy_res: float) -> bool:
        """Update residual history and check for convergence.

        Returns:
            True if converged, False otherwise.
        """
        self.p_history.append(p_res)
        self.vx_history.append(vx_res)
        self.vy_history.append(vy_res)

        if len(self.p_history) < self.m:
            return False

        def has_converged(history):
            residuals = np.array(history)
            return np.mean(residuals) >= residuals[0] * (1 - self.convergence_thresh)

        return (has_converged(self.p_history) or self.p_history[-1] < self.tol_p) and \
               (has_converged(self.vx_history) or self.vx_history[-1] < self.tol_vx) and \
               (has_converged(self.vy_history) or self.vy_history[-1] < self.tol_vy)

    def reset(self):
        self.p_history.clear()
        self.vx_history.clear()
        self.vy_history.clear()
