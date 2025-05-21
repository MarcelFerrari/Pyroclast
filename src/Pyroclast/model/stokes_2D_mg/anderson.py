import numpy as np

class AndersonAccelerator:
    def __init__(self, m, shape):
        """
        Parameters:
        - m: number of previous iterates to use
        - shape: tuple (ny1, nx1), assumed shared by vx, vy, and p
        """
        self.m = m
        self.ny1, self.nx1 = shape
        self.shape = (3, self.ny1, self.nx1)
        self.vec_size = np.prod(self.shape)

        self.xk_list = []  # Full state history (flattened)
        self.rk_list = []  # Residuals history (flattened)

    def reset(self):
        """Reset the history lists."""
        self.xk_list = []
        self.rk_list = []


    def update(self, x_k, x_next):
        """
        Apply Anderson acceleration.

        Parameters:
        - x_k: previous state, shape (3, ny1, nx1)
        - x_next: new state after one fixed-point update, shape (3, ny1, nx1)

        Returns:
        - Accelerated state (3, ny1, nx1) or None if insufficient history
        """
        # Flatten inputs
        xk = x_k.reshape(-1)
        xnp1 = x_next.reshape(-1)

        # Residual: r_k = f(x_k) - x_k
        rk = xnp1 - xk

        # Store history
        self.xk_list.append(xk.copy())
        self.rk_list.append(rk.copy())
        if len(self.xk_list) > self.m:
            self.xk_list.pop(0)
            self.rk_list.pop(0)

        if len(self.rk_list) < 2:
            return None  # Not enough history yet

        # Form residual matrix R = [r1, r2, ..., rm]
        R = np.column_stack(self.rk_list)

        try:
            G = R.T @ R
            ones = np.ones((G.shape[0], 1))
            KKT = np.block([[G, ones], [ones.T, np.zeros((1, 1))]])
            rhs = np.zeros(G.shape[0] + 1)
            rhs[-1] = 1.0

            sol = np.linalg.solve(KKT, rhs)
            alpha = sol[:-1]

            # Accelerated estimate: sum_i alpha_i * (x_i + r_i)
            x_acc = sum(a * (xk_i + rk_i) for a, xk_i, rk_i in zip(alpha, self.xk_list, self.rk_list))

            return x_acc.reshape(self.shape)

        except np.linalg.LinAlgError:
            print("Anderson acceleration failed: singular matrix")
            return None