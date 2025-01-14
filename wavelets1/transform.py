import numpy as np
import scipy
import scipy.optimize

from .wavelets import Morlet

__all__ = [
    'auto_choose_grid_params',
    'WaveletTransform'
    'grid_based_wavelet_transform',
    'inverse_grid_based_wavelet_transform',
]

def auto_choose_grid_params(data_length, sample_rate=1.0, M_C=0):
    """
    Simple heuristic picking parameters for grid-based decimation,
    plus optional negative j coverage (M_C).

    Parameters
    ----------
    data_length : int
        Number of samples in the data.
    sample_rate : float
        Sampling rate in Hz (1/dt).
    M_C : int
        Number of negative j 'compensation channels' => j_min = -M_C.

    Returns
    -------
    d, b, q, j_min, j_max, alpha_for_delta : tuple
        Default decimation factor, base scale, frequency step,
        j_min, j_max, and the golden-ratio-based shift multiplier.
    """
    # Basic time step
    dt = 1.0 / sample_rate

    # We treat 'd' as the decimation factor in seconds
    d = dt
    # Typically used scale and step
    b = 2.0
    q = 1.0

    # Negative j range for compensation channels
    j_min = -M_C
    # j_max: at least 16 or ~log2(data_length)
    j_max = max(16, int(np.ceil(np.log2(data_length))))

    # Golden ratio for Kronecker sequence
    alpha_for_delta = 1.61803

    return d, b, q, j_min, j_max, alpha_for_delta

def grid_based_wavelet_transform(
    data,
    wavelet,
    dt=1.0,
    d=None,
    b=2.0,
    q=1.0,
    j_min=0,
    j_max=128,
    alpha_for_delta=1.61803,
    xi_1=0.25,
    use_compensation=False
):
    """
    Compute the grid-based decimation wavelet transform, supporting:
      1) Negative j indices (j_min < 0).
      2) Compensation (Equation (4)) or standard mode (Equation (5)).

    Parameters
    ----------
    data : 1D ndarray
        The signal to transform.
    wavelet : wavelet object
        Must have methods .grid_time_eq4(...) or .grid_time_eq5(...).
    dt : float
        Sampling interval (1 / sample_rate).
    d : float or None
        Decimation factor in seconds. If None, defaults to dt.
    b, q : float
        Base scale, and frequency step.
    j_min, j_max : int
        Range for j: [j_min, ..., j_max - 1].
    alpha_for_delta : float
        Kronecker multiplier for δ_j = frac(alpha_for_delta * j).
    xi_1 : float
        Frequency modulation factor for eq. (5).
    use_compensation : bool
        If True => eq. (4), otherwise eq. (5).

    Returns
    -------
    W : ndarray, shape (num_l, num_j_total)
        Wavelet coefficients: W[l, j_index].
    j_values : ndarray
        The j-values actually used.
    """
    data = np.asarray(data, dtype=float)
    n_data = data.size
    
    # If d is None, default to dt so the shift is in seconds.
    if d is None:
        d = dt

    # We'll have one coefficient for each l in [0, num_l-1].
    # For simplicity: num_l = data.size
    num_l = n_data

    # Prepare j-values
    j_values = np.arange(j_min, j_max)
    num_j = len(j_values)

    # Delays: δ_j = frac(alpha_for_delta * j)
    delta_vals = np.mod(alpha_for_delta * j_values, 1.0)

    # Output array
    W = np.zeros((num_l, num_j), dtype=complex)

    # Time array in seconds
    tvals = np.arange(n_data) * dt

    for j_idx, j_val in enumerate(j_values):
        alpha_j = (1.0 / b) + (j_val / q)
        # skip invalid alpha_j
        if alpha_j <= 0:
            continue

        delta_j = delta_vals[j_idx]

        for l in range(num_l):
            # Choose eq. (4) vs eq. (5)
            if use_compensation:
                psi_l_j = wavelet.grid_time_eq4(tvals, l, j_val, d, b, q, delta_j)
            else:
                psi_l_j = wavelet.grid_time_eq5(tvals, l, j_val, d, b, q, delta_j, xi_1)

            # dot product with conjugate wavelet
            W[l, j_idx] = np.sum(data * np.conjugate(psi_l_j))

    return W, j_values

def inverse_grid_based_wavelet_transform(
    W,
    j_values,
    wavelet,
    data_length,
    dt=1.0,
    d=None,
    b=2.0,
    q=1.0,
    alpha_for_delta=1.61803,
    xi_1=0.25,
    use_compensation=False,
    normalization=1.0
):
    """
    Naïve inverse transform:
    data_approx(t) = Σ_{l,j} [W[l,j] * ψ_{l,j}(t)]

    If use_compensation=True => eq. (4), else eq. (5).
    'normalization' can be adjusted to correct amplitude.

    Parameters
    ----------
    W : ndarray, shape (num_l, num_j)
    j_values : 1D ndarray
        The j-values used in forward transform.
    wavelet : wavelet object
        Must implement the same eq4/eq5 calls as in forward transform.
    data_length : int
        Desired length of reconstructed signal.
    dt : float
        Sampling interval in seconds.
    d : float or None
        Decimation factor in seconds. If None, defaults to dt.
    b, q : float
        Same base scale and freq step from forward transform.
    alpha_for_delta : float
        Kronecker multiplier for δ_j.
    xi_1 : float
        Frequency modulation factor for eq. (5).
    use_compensation : bool
        Whether eq. (4) was used in forward transform.
    normalization : float
        Global scaling factor on the final sum.

    Returns
    -------
    data_approx : 1D ndarray (real)
    """
    if d is None:
        d = dt

    data_approx = np.zeros(data_length, dtype=complex)
    num_l, num_j = W.shape

    # time array in seconds
    tvals = np.arange(data_length) * dt

    for j_idx, j_val in enumerate(j_values):
        alpha_j = (1.0 / b) + (j_val / q)
        if alpha_j <= 0:
            continue

        delta_j = np.mod(alpha_for_delta * j_val, 1.0)

        for l in range(num_l):
            if use_compensation:
                psi_l_j_t = wavelet.grid_time_eq4(tvals, l, j_val, d, b, q, delta_j)
            else:
                psi_l_j_t = wavelet.grid_time_eq5(tvals, l, j_val, d, b, q, delta_j, xi_1)

            data_approx += W[l, j_idx] * psi_l_j_t

    return (normalization * data_approx).real

class WaveletTransform:
    """
    Grid-Based Wavelet Transform with negative j and optional compensation.
    """

    def __init__(
        self,
        data,
        wavelet=Morlet(),
        d=None,
        b=None,
        q=None,
        j_min=None,
        j_max=None,
        alpha_for_delta=None,
        xi_1=0.25,
        use_compensation=False,
        M_C=0,
        sample_rate=1.0
    ):
        """
        Parameters
        ----------
        data : 1D ndarray
        wavelet : wavelet object (must have .grid_time_eq4() & .grid_time_eq5())
        d, b, q : float or None
            Grid parameters. If None, chosen automatically.
        j_min, j_max : int or None
            Range of j. If None, chosen automatically.
        alpha_for_delta : float or None
            Kronecker multiplier for δ_j. If None, use golden ratio.
        xi_1 : float
            Phase factor for eq. (5).
        use_compensation : bool
            True => eq. (4), else eq. (5).
        M_C : int
            Number of negative j channels if j_min is not specified.
        sample_rate : float
            Used if d is chosen automatically (d=dt).
        """
        self.data = np.asarray(data)
        self.wavelet = wavelet
        self.xi_1 = xi_1
        self.use_compensation = use_compensation
        self.sample_rate = sample_rate
        self.M_C = M_C

        # Possibly auto-choose grid params
        need_auto = any(x is None for x in [d, b, q, j_min, j_max, alpha_for_delta])
        if need_auto:
            d_a, b_a, q_a, jmin_a, jmax_a, alpha_a = auto_choose_grid_params(
                data_length=len(self.data),
                sample_rate=self.sample_rate,
                M_C=self.M_C
            )
            self.d = d if d is not None else d_a
            self.b = b if b is not None else b_a
            self.q = q if q is not None else q_a
            self.j_min = j_min if j_min is not None else jmin_a
            self.j_max = j_max if j_max is not None else jmax_a
            self.alpha_for_delta = alpha_for_delta if alpha_for_delta is not None else alpha_a
        else:
            self.d = d
            self.b = b
            self.q = q
            self.j_min = j_min
            self.j_max = j_max
            self.alpha_for_delta = alpha_for_delta

        # Placeholder for storing the forward transform
        self._W = None
        self._j_values = None

    @property
    def wavelet_transform(self):
        """Compute (W, j_values) once and cache them."""
        if self._W is None:
            W, jvals = grid_based_wavelet_transform(
                data=self.data,
                wavelet=self.wavelet,
                dt=1.0 / self.sample_rate,
                d=self.d,
                b=self.b,
                q=self.q,
                j_min=self.j_min,
                j_max=self.j_max,
                alpha_for_delta=self.alpha_for_delta,
                xi_1=self.xi_1,
                use_compensation=self.use_compensation
            )
            self._W = W
            self._j_values = jvals
        return self._W

    @property
    def j_values(self):
        """Return the j-values used in the transform."""
        if self._j_values is None:
            _ = self.wavelet_transform  # forces computation
        return self._j_values

    def inverse_transform(self, normalization=1.0):
        """
        Reconstruct the signal (naïve) from the stored W[l,j].

        normalization : float
            Overall amplitude scale factor to multiply the sum.

        Returns
        -------
        data_approx : 1D ndarray (real)
        """
        if self._W is None:
            _ = self.wavelet_transform
        data_approx = inverse_grid_based_wavelet_transform(
            W=self._W,
            j_values=self._j_values,
            wavelet=self.wavelet,
            data_length=len(self.data),
            dt=1.0 / self.sample_rate,
            d=self.d,
            b=self.b,
            q=self.q,
            alpha_for_delta=self.alpha_for_delta,
            xi_1=self.xi_1,
            use_compensation=self.use_compensation,
            normalization=normalization
        )
        return data_approx

    @property
    def wavelet_power(self):
        """Magnitude-squared of the transform coefficients."""
        return np.abs(self.wavelet_transform) ** 2
    
WaveletAnalysis = WaveletTransform
