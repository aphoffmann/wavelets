import numpy as np

__all__ = ['Morlet']

class Morlet:
    """
    Complex Morlet wavelet class with optional grid-based transform support.
    """

    def __init__(self, w0=6):
        """
        Parameters
        ----------
        w0 : float
            Nondimensional frequency constant. A value around 6 is common.
        """
        self.w0 = w0
        # Predefined constant from Torrence & Compo (if w0=6).
        if w0 == 6:
            self.C_d = 0.776

    def __call__(self, t, s=1.0, complete=True):
        """
        Call the time-domain wavelet function directly.
        """
        return self.time(t, s=s, complete=complete)

    def time(self, t, s=1.0, complete=True):
        """
        Time-domain Morlet wavelet, centered at zero.
        """
        w = self.w0
        x = t / s
        output = np.exp(1j * w * x)
        if complete:
            output -= np.exp(-0.5 * (w**2))
        output *= np.exp(-0.5 * (x**2)) * np.pi**(-0.25)
        return output

    def fourier_period(self, s):
        """
        Equivalent Fourier period for the Morlet wavelet.
        """
        return 4 * np.pi * s / (self.w0 + np.sqrt(self.w0**2 + 2))

    def scale_from_period(self, period):
        """
        Compute the wavelet scale from the given Fourier period.
        """
        coeff = np.sqrt(self.w0**2 + 2)
        return period * (coeff + self.w0) / (4.0 * np.pi)

    def frequency(self, w, s=1.0):
        """
        Frequency-domain representation of the Morlet wavelet.
        """
        x = w * s
        # Simple Heaviside step: wavelet is zero for negative frequencies.
        Hw = (w > 0).astype(float)
        return (np.pi**-0.25) * Hw * np.exp(-0.5 * (x - self.w0)**2)

    def coi(self, s):
        """
        Cone of Influence scaling for the Morlet wavelet.
        """
        return np.sqrt(2) * s

    def grid_time(self, t, l, j, d, b, q, delta_j, xi_1):
        """
        Discrete grid-based wavelet for stable transforms (see Eq. (5)).

        Parameters
        ----------
        t       : array-like
                  Time index.
        l, j    : int
                  Indices in the discrete grid.
        d, b, q : float
                  Transform parameters (decimation factor, base scale, freq step).
        delta_j : float
                  Quasi-random delay for channel j.
        xi_1    : float
                  Frequency modulation factor.

        Returns
        -------
        psi_lj : array of complex
                  Sampled wavelet function at each time point.
        """
        # Shift & scale
        shift = d * (l + delta_j)
        tau = (t - shift) / b
        base_val = self.time(tau, s=1.0, complete=True)
        phase_factor = np.exp(2j * np.pi * xi_1 * j * (t - shift) / q)
        return (1.0 / np.sqrt(b)) * base_val * phase_factor


    def grid_time_eq4(self, t, l, j, d, b, q, delta_j):
        # alpha_j = 1/b + j/q
        alpha_j = (1.0 / b) + (j / q)
        shift = d * (l + delta_j)
        # argument
        arg = alpha_j * (t - shift)
        base = self.time(arg, s=1.0, complete=True)
        # amplitude factor
        return np.sqrt(alpha_j) * base

    def grid_time_eq5(self, t, l, j, d, b, q, delta_j, xi_1):
        # eq. (5)
        shift = d * (l + delta_j)
        tau = (t - shift) / b
        base = self.time(tau, s=1.0, complete=True)
        phase_factor = np.exp(2j * np.pi * xi_1 * j * (t - shift) / q)
        return (1.0 / np.sqrt(b)) * base * phase_factor