"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: anderson.py
Description: Implementation of Anderson Acceleration (AA-II / Pulay) with
             optional Tikhonov regularization for fixed-point solvers.
                    
Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np
from Pyroclast.logging import get_logger

logger = get_logger(__name__)

class _AAParams:
    def __init__(self, ctx):
        """
        Reads Anderson Acceleration parameters from the context object.

        Parameters
        ----------
        ctx : Context
            Context object containing input parameters.
        """
        s, p, o = ctx
        self.enabled = bool(p.get("enable_anderson", True))
        self.m = int(p.get("anderson_m", 15))
        self.beta = float(p.get("anderson_beta", 0.7))
        self.reg = float(p.get("anderson_reg", 0.0))
        self.scale_reg = bool(p.get("anderson_scale_reg", False))

class AndersonAccelerator:
    def __init__(self, ctx, shape, dtype=np.float64):
        """
        AA-II (Pulay) with optional Tikhonov regularization.
        Allocation is done up-front from `shape` (e.g., (3, ny1, nx1)).

        Parameters
        ----------
        m : int
            History depth.
        shape : tuple[int, ...]
            Shape of the state vector/array (e.g., (3, ny, nx)).
        beta : float
            Damping (1.0 = standard AA; <1.0 under-relaxed).
        reg : float
            Tikhonov coefficient; 0.0 disables regularization.
        scale_reg : bool
            If True, scales reg by trace(G)/n for magnitude invariance.
        dtype : numpy dtype
            Buffer dtype (defaults to float64).
        """
        params = _AAParams(ctx)
        self.enabled = params.enabled
        
        # If not enabled, do nothing
        if not self.enabled:
            return
                
        self.m = params.m
        self.beta = params.beta
        self.reg = params.reg
        self.scale_reg = params.scale_reg
        
        assert self.m >= 1, "Anderson m must be at least 1"

        self.shape = tuple(shape)
        self.vec_size = int(np.prod(self.shape))
        self.dtype = dtype

        # ring buffers (vec_size × m)
        self.X  = np.zeros((self.vec_size, self.m), dtype=self.dtype)
        self.FX = np.zeros((self.vec_size, self.m), dtype=self.dtype)
        self.R  = np.zeros((self.vec_size, self.m), dtype=self.dtype)

        self.k = 0  # total updates made

    def check_enabled(method):
        def wrapper(self, *args, **kwargs):
            if not getattr(self, "enabled", True):
                return None
            return method(self, *args, **kwargs)
        return wrapper

    @check_enabled
    def reset(self):
        """Zero history (keeps allocations)."""    
        self.X.fill(0)
        self.FX.fill(0)
        self.R.fill(0)
        self.k = 0

    @check_enabled
    def update(self, xk, fxk) -> np.ndarray | None:
        """
        Provide xk and its mapped iterate fxk = G(x_k).
        Returns accelerated x with shape == self.shape, or None until enough history.

        xk, fxk are expected to be flat 1D arrays of size vec_size.
        Returns None if something goes wrong (e.g., not enough history) and a flat array otherwise.
        """

        # Assert that arrays are flat and of correct size
        assert xk.ndim == 1 and fxk.ndim == 1, "xk and fxk must be flat 1D arrays"
        assert xk.size == self.vec_size, f"xk size {xk.size} != expected {self.vec_size}"
        assert fxk.size == self.vec_size, f"fxk size {fxk.size} != expected {self.vec_size}"

        col = self.k % self.m
        self.X[:,  col] = xk
        self.FX[:, col] = fxk
        self.R[:,  col] = fxk - xk

        self.k += 1
        n = min(self.k, self.m)
        if n < 2:
            return None  # need at least two history points

        # active window (vec_size × n)
        if n < self.m:
            R_sub, X_sub, FX_sub = self.R[:, :n], self.X[:, :n], self.FX[:, :n]
        else:
            R_sub, X_sub, FX_sub = self.R, self.X, self.FX

        # KKT for: min ||R_sub α||^2 s.t. 1^T α = 1
        G = R_sub.T @ R_sub  # (n, n)
        if self.reg > 0.0:
            lam = self.reg
            if self.scale_reg:
                tr = float(np.trace(G))
                lam *= (tr / n) if tr > 0.0 else 1.0
            G = G + lam * np.eye(n, dtype=G.dtype)

        ones = np.ones((n, 1), dtype=G.dtype)
        KKT  = np.block([[G,      ones],
                         [ones.T, np.zeros((1, 1), dtype=G.dtype)]])
        rhs  = np.zeros(n + 1, dtype=G.dtype)
        rhs[-1] = 1.0

        try:
            sol = np.linalg.solve(KKT, rhs)
        except np.linalg.LinAlgError:
            logger.warning("Singular KKT system in Anderson Acceleration")
            return None

        alpha = sol[:-1]  # (n,)

        x_bar  = X_sub  @ alpha
        fx_bar = FX_sub @ alpha
        x_acc  = (1.0 - self.beta) * x_bar + self.beta * fx_bar
        return x_acc