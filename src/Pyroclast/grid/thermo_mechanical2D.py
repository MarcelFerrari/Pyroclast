import numpy as np
from Pyroclast.grid.staggered2D import BasicStaggered2D
from Pyroclast.interpolation.linear_2D_cpu \
    import interpolate_markers2grid as interpolate

class ThermoMechanicalGrid2D(BasicStaggered2D):
    def __init__(self, ctx):
        """
        Initialization method for the thermo-mechanical 2D grid.

        This method extends the BasicStaggered2D grid by adding temperature nodes.
        """

        # First, initialize the basic staggered grid
        super().__init__(ctx)

        s, p, o = ctx

        # Create the temperature nodes on the pressure grid
        s.xT = s.xp
        s.yT = s.yp

    def interpolate(self, ctx):
        """
        Interpolate density and viscosity from markers to grid nodes.
        """
        # First interpolate material properties using the base class method
        super().interpolate(ctx)
        
        s, p, o = ctx


        # % Interpolate marker thermal properties to grid;
        # Alpha = interpolate_markers(px, py, xm, ym, alpham);

        # Thermal conductivity on vx nodes, W/mK
        s.kvx = interpolate(s.xvx,                      # x positions of vx nodes
                          s.yvx,                      # y positions of vx nodes
                          s.xm,                       # Marker x positions
                          s.ym,                       # Marker y positions
                          s.km,                       # Marker thermal conductivity
                          indexing="equidistant",     # Equidistant grid spacing
                          return_weights=False)       # Do not return weights
        
        # Thermal conductivity on vy nodes, W/mK
        s.kvy = interpolate(s.xvy,                      # x positions of vy nodes
                          s.yvy,                      # y positions of vy nodes
                          s.xm,                       # Marker x positions
                          s.ym,                       # Marker y positions
                          s.km,                       # Marker thermal conductivity
                          indexing="equidistant",     # Equidistant grid spacing
                          return_weights=False)       # Do not return weights
        
        s.rhocp = interpolate(s.xT,                       # x positions of temperature nodes
                            s.yT,                       # y positions of temperature nodes
                            s.xm,                       # Marker x positions
                            s.ym,                       # Marker y positions
                            s.rhocpm,                   # Marker density * specific heat
                            indexing="equidistant",     # Equidistant grid spacing
                            return_weights=False)       # Do not return weights
        
        # rhocp weighted interpolation of temperature
        s.T0 = interpolate(s.xT,                           # x positions of temperature nodes
                        s.yT,                           # y positions of temperature nodes
                        s.xm,                           # Marker x positions
                        s.ym,                           # Marker y positions
                        s.Tm * s.rhocpm,                # Marker temperature * density * specific heat
                        indexing="equidistant",         # Equidistant grid spacing
                        return_weights=False)           # Do not return weights
        s.T0 /= s.rhocp  # Divide to get temperature

        # Thermal expansivity on temperature nodes, 1/K
        s.alpha = interpolate(s.xT,                       # x positions of temperature nodes
                            s.yT,                       # y positions of temperature nodes
                            s.xm,                       # Marker x positions
                            s.ym,                       # Marker y positions
                            s.alpham,                   # Marker thermal expansivity
                            indexing="equidistant",     # Equidistant grid spacing
                            return_weights=False)       # Do not return weights


        # # Store interpolate values to context
        # mask = np.isfinite(kvx)
        # s.kvx[mask] = kvx[mask]

        # mask = np.isfinite(kvy)
        # s.kvy[mask] = kvy[mask]

        # mask = np.isfinite(rhocp)
        # s.rhocp[mask] = rhocp[mask]

        # mask = np.isfinite(T0)
        # s.T0[mask] = T0[mask]

        # mask = np.isfinite(alpha)
        # s.alpha[mask] = alpha[mask]