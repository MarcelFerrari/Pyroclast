import numpy as np
import numba as nb
from Pyroclast.pool.inclusion2D import CircularInclusion

class CircularInclusionThermoMechanical(CircularInclusion):
    """
    Circular inclusion in 2D.
    """
    def __init__(self, ctx):
        # Initialize markers for mechanical problem
        super().__init__(ctx)

        # Read context
        s, p, o = ctx

        # Marker material properties
        s.km = np.zeros(s.nm, dtype=np.float64)     # Thermal conductivity, W/mK
        s.rhocpm = np.zeros(s.nm, dtype=np.float64) # Volumetric heat capacity, J/m^3K
        s.Tm = np.zeros(s.nm, dtype=np.float64)     # Temperature, K
        s.alpham = np.zeros(s.nm, dtype=np.float64) # Thermal expansivity, 1/K

        _init_circular_thermomechanical_inclusion(s.nmx, s.nmy,
                                s.dxm, s.dym,
                                self.x0, self.y0, self.r,
                                s.xm, s.ym,
                                s.rhom,
                                s.km,
                                s.rhocpm,
                                s.Tm,
                                s.alpham)
                              

@nb.njit(cache=True)
def _init_circular_thermomechanical_inclusion(nmx, nmy,
                                        dxm, dym,
                                        x0, y0, r,
                                        xm, ym,
                                        rhom,
                                        km,
                                        rhocpm,
                                        Tm,
                                        alpham):
    """
    Initialize the material properties of the markers.

    nmx: Number of markers in x direction
    nmy: Number of markers in y direction
    dxm: Marker spacing in x direction
    dym: Marker spacing in y direction
    x0: x coordinate of the center of the circle
    y0: y coordinate of the center of the circle
    r: radius of the circle
    xm: Marker positions in x direction
    ym: Marker positions in y direction
    rhom: Marker density array to be filled
    km: Marker thermal conductivity array to be filled
    rhocpm: Marker volumetric heat capacity array to be filled
    Tm: Marker temperature array to be filled
    alpham: Marker thermal expansivity array to be filled
    """

    m = 0
    # Initialize marker values
    for i in range(nmy):
        for j in range(nmx):
            # Read marker position
            xm0 = xm[m]
            ym0 = ym[m]

            # Set up material properties of markers
            d = (xm0 - x0)**2 + (ym0 - y0)**2

            # Inside of circle
            if d < r**2:
                cp = 1100.0  # Specific heat, J/kgK
                rho = rhom[m]
                rhocpm[m] = rho * cp
                km[m] = 2.0  # Thermal conductivity, W/mK
                Tm[m] = 1773.0  # Temperature, K
                alpham[m] = 3e-5  # Thermal expansivity, 1/K
            else: # Outside of circle
                cp = 900.0  # Specific heat, J/kgK
                rho = rhom[m]
                rhocpm[m] = rho * cp
                km[m] = 3.0  # Thermal conductivity, W/mK
                Tm[m] = 1573.0  # Temperature, K
                alpham[m] = 2e-5  # Thermal expansivity, 1/K

            m += 1