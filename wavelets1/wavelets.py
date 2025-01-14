import numpy as np

__all__ = ['Morlet']

class Morlet:
    """
    Complex Morlet wavelet class with optional grid-based transform support.
    """

    def __init__(self, w0=6):
        """
        w0 : float
            Nondimensional frequency constant (~6 recommended).
        """
        self.w0 = w0
        if w0 == 6:
            self.C_d = 0.776

    def __call__(self, t, s=1.0, complete=True):
        return self.time(t, s=s, complete=complete)

    def time(self, t, s=1.0, complete=True):
        """
        Time-domain Morlet wavelet, centered at zero.
        t is in seconds; s is dimensionless scale.
        """
        w = self.w0
        x = t / s
        out = np.exp(1j * w * x)
        if complete:
            out -= np.exp(-0.5 * (w**2))
        out *= np.exp(-0.5 * (x**2)) * np.pi**(-0.25)
        return out

    def grid_time_eq4(self, t, l, j, d, b, q, delta_j):
        """
        Eq. (4):
        psi_{l,j}(t) = sqrt(1/b + j/q)* psi( (1/b + j/q)*(t - d(l+δ_j)) ).
        """
        alpha_j = (1.0 / b) + (j / q)
        shift = d * (l + delta_j)
        arg = alpha_j * (t - shift)
        base = self.time(arg, s=1.0, complete=True)
        return np.sqrt(alpha_j) * base

    def grid_time_eq5(self, t, l, j, d, b, q, delta_j, xi_1):
        """
        Eq. (5):
          psi_{l,j}(t) = (1/sqrt(b)) * psi((t - d(l+δ_j))/b)
                         * exp(2π i xi_1 * j*(t - d(l+δ_j)) / q).
        """
        shift = d * (l + delta_j)
        tau = (t - shift) / b
        base = self.time(tau, s=1.0, complete=True)
        phase = np.exp(
            2j * np.pi * xi_1 * j * (t - shift) / q
        )
        return (1.0 / np.sqrt(b)) * base * phase
