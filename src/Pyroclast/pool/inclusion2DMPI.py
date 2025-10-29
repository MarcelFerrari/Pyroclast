import numpy as np
import numba as nb

from Pyroclast.pool.basic2DMPI import Basic2DStokesMPI
from Pyroclast.utils import wrap_periodic

class RectangularInclusionMPI(Basic2DStokesMPI):
    """
    Circular inclusion in 2D.
    """
    def __init__(self, ctx):
        # Initialize markers
        super().__init__(ctx)

        # Read context
        s, p, o = ctx

        # Define parameters for a circular inclusion
        x0 = p.xsize_global/2. # x coordinate of the center of the rectangle
        y0 = p.ysize_global/2. # y coordinate of the center of the rectangle
        h = p.h # height of the rectangle
        w = p.w # width of the rectangle
        
        # Marker material properties
        s.rhom = np.zeros(s.nm, dtype=np.float64)
        s.etam = np.zeros(s.nm, dtype=np.float64)


        s.rhom, s.etam = _init_rectangular_inclusion(s.nmx, s.nmy,
                                                     s.m_xmin, s.m_ymin,
                                                     s.dxm, s.dym,
                                                     x0, y0, w, h,
                                                     s.xm, s.ym,
                                                     p.rho_plume,
                                                     p.rho_mantle,
                                                     p.eta_plume,
                                                     p.eta_mantle,
                                                     s.rhom, s.etam)
        
        # Apply periodic boundary conditions to marker positions
        s.xm = wrap_periodic(s.xm, s.m_xmin, s.m_xmax)
        s.ym = wrap_periodic(s.ym, s.m_ymin, s.m_ymax)
      
@nb.njit(cache=True)
def _init_rectangular_inclusion(nmx, nmy, xmin, ymin, dxm, dym, x0, y0, w, h, xm, ym, rho_plume, rho_mantle, eta_plume, eta_mantle, rhom, etam):
    """
    Initialize the material properties of the markers.

    nmx: Number of markers in x direction
    nmy: Number of markers in y direction
    dxm: Marker spacing in x direction
    dym: Marker spacing in y direction
    x0: x coordinate of the center of the rectangle
    y0: y coordinate of the center of the rectangle
    w: width of the rectangle
    h: height of the rectangle
    xm: Marker positions in x direction
    ym: Marker positions in y direction
    rhom: Marker density
    etam: Marker viscosity
    """

    m = 0
    # Initialize marker values
    for i in range(nmy):
        for j in range(nmx):
            # Compute marker index
            xm[m] = dxm/2 + j*dxm + (np.random.uniform()-0.5)*dxm + xmin
            ym[m] = dym/2 + i*dym + (np.random.uniform()-0.5)*dym + ymin

            # Set up material properties of markers
            if (xm[m] > x0 - w/2 and xm[m] < x0 + w/2 and
                ym[m] > y0 - h/2 and ym[m] < y0 + h/2):
                rhom[m] = rho_plume
                etam[m] = eta_plume
            else:
                rhom[m] = rho_mantle
                etam[m] = eta_mantle

            m += 1
    return rhom, etam