"""
Utilities for optional viscosity rescaling during the multigrid solve.
"""

import numpy as np
import numba as nb
from Pyroclast.logging import get_logger

logger = get_logger(__name__)


class ViscosityRescaler:
    """Manage viscosity rescaling for a grid hierarchy."""

    def __init__(self, ctx, hierarchy):
        s, p, o = ctx
        self.hierarchy = hierarchy
        self.enable = p.get("eta_scaling", False)

        # If no scaling, skip
        if not self.enable:
            return
        
        # Store reference to original viscosity
        self.stokes_etab = s.etab
        self.stokes_etap = s.etap

        # Ensure that shapes match
        assert self.stokes_etab.shape == self.etab_comp.shape
        assert self.stokes_etap.shape == self.etap_comp.shape

        # Rescaling parameters
        self.cycle_count = 0
        self.rescale_count = 0
        self.rescale_interval = p.eta_cycle_interval
        self.total_rescales = p.eta_ncycles
        self.progress = 0.0
        self.progress_step = 1.0 / max(self.total_rescales - 1, 1)

    def reset(self):
        """Reset the rescaler state."""
        self.cycle_count = 0
        self.rescale_count = 0
        self.progress = 0.0
        self.etab_min = 0.0
        self.etap_min = 0.0
        self.stokes_etab = None
        self.stokes_etap = None

    def set(self, etab, etap):
        """Set new reference viscosity fields."""
        if not self.enable:
            return
        self.stokes_etab = etab
        self.stokes_etap = etap
        self.etab_min = np.min(self.stokes_etab[:-1, :-1])
        self.etap_min = np.min(self.stokes_etap[:-1, :-1])
        # Apply scaling immediately
        self._apply_scaling()
        self._propagate()

    @property
    def etab_comp(self):
        """Computational Viscosity in basic Nodes"""
        return self.hierarchy[0].etab
    
    @property
    def etap_comp(self):
        """Computational Viscosity in pressure Nodes"""
        return self.hierarchy[0].etap

    def done_rescaling(self):
        """Check whether all planned rescaling cycles are done."""
        if not self.enable:
            return True
        return self.rescale_count >= self.total_rescales

    def _propagate(self):
        """Propagate viscosity values down the grid hierarchy."""
        for lvl in range(1, len(self.hierarchy)):
            prev = self.hierarchy[lvl - 1]
            curr = self.hierarchy[lvl]
            curr.restrict_properties(prev)

    def _apply_scaling(self):
        """Apply viscosity rescaling to the finest level."""
        if not self.enable:
            return
        theta = min(self.progress, 1.0)
        fine = self.hierarchy[0]
        _interpolate_viscosity(fine.nx1, fine.ny1, theta,
                               self.etab_min, self.etap_min,
                               self.stokes_etab, self.stokes_etap,
                               self.etab_comp, self.etap_comp)

    def update_viscosity(self):
        """Possibly rescale viscosity based on current cycle count."""
        if not self.enable:
            return False

        self.cycle_count += 1
        if self.cycle_count < self.rescale_interval:
            return False
        if self.rescale_count >= self.total_rescales:
            return False

        logger.info(
            f"Rescaling viscosity: cycle {self.cycle_count}, "
            f"rescale {self.rescale_count + 1}/{self.total_rescales}, "
            f"progress {self.progress:.2f}")

        self.progress = min(1.0, self.progress + self.progress_step)
        self._apply_scaling()
        self._propagate()

        self.cycle_count = 0
        self.rescale_count += 1
        return True

@nb.njit(parallel=True, cache=True)
def _interpolate_viscosity(nx1, ny1, theta, etab_min, etap_min, etab, etap, etab_comp, etap_comp):
    for i in nb.prange(ny1):
        for j in nb.prange(nx1):
            etab_comp[i, j] = (1.0 - theta) * etab_min + theta * etab[i, j]
            etap_comp[i, j] = (1.0 - theta) * etap_min + theta * etap[i, j]