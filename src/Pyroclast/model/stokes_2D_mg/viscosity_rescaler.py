"""
Utilities for optional viscosity rescaling during the multigrid solve.
"""

import numpy as np


class ViscosityRescaler:
    """Manage viscosity rescaling for a grid hierarchy."""

    def __init__(self, ctx, hierarchy):
        state, params, _opts = ctx
        self.hierarchy = hierarchy
        self.enable = params.get("eta_scaling", False)
        if not self.enable:
            return

        self.orig_etab = state.etab
        self.orig_etap = state.etap
        self.cycle_count = 0
        self.rescale_count = 0
        self.rescale_interval = params.eta_cycle_interval
        self.total_rescales = params.eta_ncycles
        self.progress = 0.0
        self.progress_step = 1.0 / max(self.total_rescales - 1, 1)
        self.etab_min = np.nanmin(self.orig_etab[:-1, :-1])
        self.etap_min = np.nanmin(self.orig_etap[:-1, :-1])

        self._apply_scaling()
        self._propagate()

    def done_rescaling(self):
        if not self.enable:
            return True
        return self.rescale_count >= self.total_rescales

    def _propagate(self):
        for lvl in range(1, len(self.hierarchy)):
            prev = self.hierarchy[lvl - 1]
            coarse = self.hierarchy[lvl]
            coarse.restrict_properties(prev)

    def _apply_scaling(self):
        if not self.enable:
            return
        theta = min(self.progress, 1.0)
        fine = self.hierarchy[0]
        fine.etab = self.etab_min * (1.0 - theta) + self.orig_etab * theta
        fine.etap = self.etap_min * (1.0 - theta) + self.orig_etap * theta
        fine.etab = np.nan_to_num(fine.etab, copy=False)
        fine.etap = np.nan_to_num(fine.etap, copy=False)

    def update_viscosity(self):
        if not self.enable:
            return False

        self.cycle_count += 1
        if self.cycle_count < self.rescale_interval:
            return False
        if self.rescale_count >= self.total_rescales:
            return False

        print(f"Rescaling viscosity: cycle {self.cycle_count}, "
              f"rescale {self.rescale_count + 1}/{self.total_rescales}, "
              f"progress {self.progress:.2f}")
        
        self.progress = min(1.0, self.progress + self.progress_step)
        self._apply_scaling()
        self._propagate()

        self.cycle_count = 0
        self.rescale_count += 1
        return True

