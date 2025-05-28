import numpy as np

class AndersonAccelerator:
    def __init__(self, m, shape, beta=0.7):
        """
        Pre-allocate all buffers of size (vec_size, m).
        """
        self.m = m
        self.beta = beta
        self.ny1, self.nx1 = shape
        self.shape = (3, self.ny1, self.nx1)
        self.vec_size = 3 * self.ny1 * self.nx1

        # cyclic buffers
        self.X  = np.zeros((self.vec_size, m))
        self.FX = np.zeros((self.vec_size, m))
        self.R  = np.zeros((self.vec_size, m))

        # counter of total updates, effective window size
        self.k = 0

    def reset(self):
        """Clear history back to zero."""
        self.X.fill(0)
        self.FX.fill(0)
        self.R.fill(0)
        self.k = 0

    def update(self, x_k, x_next):
        xk  = x_k.reshape(-1)
        fxk = x_next.reshape(-1)

        col = self.k % self.m
        # overwrite oldest entry in cyclic fashion
        self.X[:,  col] = xk
        self.FX[:, col] = fxk
        self.R[:,  col] = fxk - xk

        self.k += 1
        n = min(self.k, self.m)

        if n < 2:
            return None  # need at least two history points

        # pick residual matrix (and X/FX) of size vec_size × n
        if n < self.m:
            R_sub  = self.R[:,  :n]
            X_sub  = self.X[:,  :n]
            FX_sub = self.FX[:, :n]
        else:
            # full buffer when we've filled m slots
            R_sub  = self.R
            X_sub  = self.X
            FX_sub = self.FX

        # build and solve the small KKT system
        G    = R_sub.T @ R_sub               # (n,n)
        ones = np.ones((n, 1))
        KKT  = np.block([[G,      ones],
                         [ones.T, np.zeros((1, 1))]])
        rhs  = np.zeros(n + 1)
        rhs[-1] = 1.0

        try:
            sol   = np.linalg.solve(KKT, rhs)
            alpha = sol[:-1]   # length-n

            # form accelerated update
            x_bar  = X_sub  @ alpha
            fx_bar = FX_sub @ alpha
            x_acc  = (1-self.beta)*x_bar + self.beta*fx_bar

            return x_acc.reshape(self.shape)

        except np.linalg.LinAlgError:
            print("Singular KKT matrix in Anderson update")
            return None
