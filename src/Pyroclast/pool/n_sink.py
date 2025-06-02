"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: NSink.py
Description: This file implements 2D marker pool for 2D N-sink sedimentation benchmark.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np
import numba as nb
from scipy.stats import qmc


from Pyroclast.pool.basic2D import RK42DStokes

class NSink(RK42DStokes):
    """
    Circular inclusion in 2D.
    """
    def __init__(self, ctx):
        # Initialize markers
        super().__init__(ctx)

        # Read context
        s, p, o = ctx

        # Define parameters for a circular inclusion
        n_sediments = p.n_sediments # number of sediments
        r = p.r # radius of the sediment
        r_dev = p.r_dev # Allowed radius deviation (+-)

        
        # Generate n random x0 and y0 that are safely inside the domain
        # and have at least 5% xsize and ysize distance from the edges

        # Create a Sobol sequence sampler in 2D
        try:
            # Try new scipy.stats.qmc.Sobol API
            rng = np.random.default_rng(o.seed)
            sampler = qmc.Sobol(d=2, scramble=True, rng=rng)
        except:
            # Fallback for older versions of scipy
            # that do not support the new API
            sampler = qmc.Sobol(d=2, scramble=True, seed=o.seed)

        # Generate n samples in the unit square [0, 1]^2
        samples = sampler.random(n=n_sediments)

        # Map from unit square to your desired range
        x0 = 0.05*p.xsize + (0.90*p.xsize - 2*r*(1.0 + r_dev)) * samples[:, 0]
        y0 = 0.05*p.ysize + (0.90*p.ysize - 2*r*(1.0 + r_dev)) * samples[:, 1]

        # Generate random radii
        r = np.random.uniform(r*(1.0 - r_dev), r*(1.0 + r_dev), n_sediments)
        
        # Marker material properties
        s.rhom = np.zeros(s.nm, dtype=np.float64)
        s.etam = np.zeros(s.nm, dtype=np.float64)


        s.rhom, s.etam = _init_n_sink_inclusion(s.nmx, s.nmy,
                                                s.dxm, s.dym,
                                                x0, y0, r,
                                                s.xm, s.ym,
                                                p.rho_plume,
                                                p.rho_mantle,
                                                p.eta_plume,
                                                p.eta_mantle,
                                                s.rhom, s.etam)

@nb.njit(cache=True)
def _init_n_sink_inclusion(nmx, nmy, dxm, dym, x0, y0, r, xm, ym, rho_plume, rho_mantle, eta_plume, eta_mantle, rhom, etam):
    """
    Initialize the material properties of the markers.

    nmx: Number of markers in x direction
    nmy: Number of markers in y direction
    dxm: Marker spacing in x direction
    dym: Marker spacing in y direction
    x0: array of x coordinate of the centers of the sediments
    y0: array of y coordinate of the centers of the sediments
    r: array of radii of the sediments
    xm: Marker positions in x direction
    ym: Marker positions in y direction
    rhom: Marker density
    etam: Marker viscosity
    """

    m = 0
    # Initialize marker values
    for i in range(nmy):
        for j in range(nmx):
            # Compute marker coordinates
            xm[m] = dxm/2 + j*dxm + (np.random.uniform()-0.5)*dxm
            ym[m] = dym/2 + i*dym + (np.random.uniform()-0.5)*dym

            # Compute distance from all centers of the N sediments
            d = (xm[m] - x0)**2 + (ym[m] - y0)**2

            # Check if the marker is inside any sediment
            # If the distance is less than the radius, set the marker properties
            # to the sediment properties
            if np.any(d < r**2):
                rhom[m] = rho_plume
                etam[m] = eta_plume
            else:
                rhom[m] = rho_mantle
                etam[m] = eta_mantle

            m += 1
    return rhom, etam
