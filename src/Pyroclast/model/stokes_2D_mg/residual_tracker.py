"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: residual_tracker.py
Description: This file implements a residual tracker to monitor the convergence of
             the Stokes residuasls.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np
from collections import deque

class ResidualTracker:
    def __init__(self, m, tol, convergence_thresh = 1e-3):
        """
        Args:
            m: Number of recent residuals to track.
            tol: Convergence tolerance.
            convergence_thresh: Relative change threshold to detect convergence.
        """
        self.m = m
        self.tol = tol
        self.convergence_thresh = convergence_thresh
        self.history = {
            'p': deque(maxlen=m),
            'vx': deque(maxlen=m),
            'vy': deque(maxlen=m)
        }

    def update(self, p_res: float, vx_res: float, vy_res: float) -> bool:
        """Update residual history and check for convergence.
        
        Returns:
            True if converged or converged, False otherwise.
        """
        self.history['p'].append(p_res)
        self.history['vx'].append(vx_res)
        self.history['vy'].append(vy_res)

        # Check if the residuals fall below tolerance
        if max(p_res, vx_res, vy_res) < self.tol:
            return True

        # If we have fewer than m values, we can't assess convergence
        if len(self.history['p']) < self.m:
            return False

        # Check convergence for each component
        converged = True
        for key, hist in self.history.items():
            residuals = np.array(hist)
            if np.min(residuals) < residuals[0] * (1 - self.convergence_thresh):
                converged = False
                break
        
        return converged

    def reset(self):
        for key in self.history:
            self.history[key].clear()
