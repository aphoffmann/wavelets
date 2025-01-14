import numpy as np
import scipy
import scipy.optimize

from .wavelets import Morlet

__all__ = [
    'WaveletTransform',
    'WaveletAnalysis',
    'grid_based_wavelet_transform',
    'inverse_grid_based_wavelet_transform',
    'auto_choose_grid_params'
]


def grid_based_wavelet_transform(
    data,
    wavelet,
    d=1.0,
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
        Must have .grid_time_eq4(...) and .grid_time_eq5(...) methods 
        (or a single method that can handle both).
    d, b, q : float
        Decimation factor, base scale, and frequency step.
    j_min, j_max : int
        Range of j. The code will loop j = [j_min, j_min+1, ..., j_max-1].
    alpha_for_delta : float
        Kronecker multiplier for channel-specific delay, δ_j = frac(alpha_for_delta * j).
    xi_1 : float
        Frequency modulation factor used if use_compensation=False (Eq. (5)).
    use_compensation : bool
        If True, use Eq. (4): 
            ψ_{l,j}(t) = sqrt(1/b + j/q) * ψ((1/b + j/q)*(t - d(l+δ_j))).
        If False, use Eq. (5):
            ψ_{l,j}(t) = (1/sqrt(b)) * ψ((t - d(l+δ_j))/b) * exp(2π i xi_1 * j*(t - ...)/q).

    Returns
    -------
    W : 2D ndarray of shape (num_l, num_j_total)
        Wavelet coefficients: W[l, j_index], 
        where j_index runs from 0..(j_max - j_min - 1).
    j_values : ndarray
        The actual j-values used (from j_min..j_max-1).
    """
    data = np.asarray(data)
    n_data = data.size

    # We'll interpret l in [0..num_l-1], but we won't fix it here:
    # Instead, let num_l match the data length or something else.
    # For demonstration, let:
    num_l = len(data)

    # Construct array of j values, e.g. j_min=-2, j_max=128
    j_values = np.arange(j_min, j_max)
    num_j = len(j_values)

    # Precompute the delays for each j
    delta_vals = np.mod(alpha_for_delta * j_values, 1.0)

    # Prepare the output
    W = np.zeros((num_l, num_j), dtype=complex)

    tvals = np.arange(n_data)

    for j_idx, j_val in enumerate(j_values):
        # For negative j, eq. (3) still says s_j = 1 / (b^-1 + q^-1 * j_val)
        # or eq. (4) has alpha_j = (1/b + j_val/q).
        # If this is <= 0, skip or zero out (no wavelet).
        alpha_j = (1.0 / b) + (j_val / q)

        # Avoid alpha_j <= 0 => no meaningful wavelet scale.
        if alpha_j <= 0:
            continue

        delta_j = delta_vals[j_idx]

        for l in range(num_l):
            if use_compensation:
                # Use eq. (4) style wavelet
                psi_l_j = wavelet.grid_time_eq4(
                    tvals, l, j_val, d, b, q, delta_j
                )
            else:
                # Use eq. (5) style wavelet
                psi_l_j = wavelet.grid_time_eq5(
                    tvals, l, j_val, d, b, q, delta_j, xi_1
                )
            W[l, j_idx] = np.sum(data * np.conjugate(psi_l_j))

    return W, j_values


def inverse_grid_based_wavelet_transform(
    W,
    j_values,
    wavelet,
    data_length,
    d=1.0,
    b=2.0,
    q=1.0,
    alpha_for_delta=1.61803,
    xi_1=0.25,
    use_compensation=False,
    normalization=1.0
):
    """
    Naïve inverse, summing W[l, j] * ψ_{l,j}(t) over l,j.

    If use_compensation=True, uses Eq. (4). If False, uses Eq. (5).
    Adjust 'normalization' to tune amplitude.

    Parameters
    ----------
    W : ndarray of shape (num_l, num_j)
    j_values : ndarray
        The actual j-values used in the forward transform.
    wavelet : wavelet object
        Must have the same .grid_time_eq4 or eq5 used in forward transform.
    data_length : int
        Length of the reconstructed signal.
    d, b, q : float
    alpha_for_delta : float
    xi_1 : float
    use_compensation : bool
        Use Eq. (4) vs Eq. (5).
    normalization : float
        Global factor for amplitude.

    Returns
    -------
    data_approx : 1D ndarray (real)
    """
    data_approx = np.zeros(data_length, dtype=complex)
    num_l, num_j = W.shape
    tvals = np.arange(data_length)

    for j_idx, j_val in enumerate(j_values):
        alpha_j = (1.0 / b) + (j_val / q)
        # skip invalid alpha_j
        if alpha_j <= 0:
            continue

        delta_j = np.mod(alpha_for_delta * j_val, 1.0)

        for l in range(num_l):
            if use_compensation:
                psi_l_j_t = wavelet.grid_time_eq4(
                    tvals, l, j_val, d, b, q, delta_j
                )
            else:
                psi_l_j_t = wavelet.grid_time_eq5(
                    tvals, l, j_val, d, b, q, delta_j, xi_1
                )
            data_approx += W[l, j_idx] * psi_l_j_t

    data_approx *= normalization
    return data_approx.real


def auto_choose_grid_params(
    data_length, sample_rate=1.0, M_C=0
):
    """
    Simple heuristic picking parameters for grid-based decimation,
    plus optional negative j coverage (M_C).

    Parameters
    ----------
    data_length : int
    sample_rate : float
    M_C : int
        Number of 'compensation channels' => negative j from -M_C..-1.

    Returns
    -------
    d, b, q, j_min, j_max, alpha_for_delta
    """
    dt = 1.0 / sample_rate

    d = dt
    b = 2.0
    q = 1.0

    # j_min negative? For compensation channels, e.g. -M_C
    j_min = -M_C
    # j_max e.g. ~ log2(...) or a fixed default. We'll choose:
    j_max = max(16, int(np.ceil(np.log2(data_length))))

    alpha_for_delta = 1.61803

    return d, b, q, j_min, j_max, alpha_for_delta


class WaveletTransform:
    """
    Grid-Based Wavelet Transform with negative j and optional compensation.
    """

    def __init__(self, data,
                 wavelet=Morlet(),
                 d=None, b=None, q=None,
                 j_min=None, j_max=None,
                 alpha_for_delta=None,
                 xi_1=0.25,
                 use_compensation=False,
                 M_C=0,
                 sample_rate=1.0):
        """
        Parameters
        ----------
        data : 1D ndarray
        wavelet : wavelet object
            Must implement grid_time_eq4(...) and grid_time_eq5(...).
        d, b, q : float
            Grid parameters. If None, chosen automatically.
        j_min, j_max : int
            Range for j. If None, chosen automatically.
        alpha_for_delta : float
            Kronecker multiplier for δ_j. If None, defaults to golden ratio.
        xi_1 : float
            Phase factor for eq. (5).
        use_compensation : bool
            If True, use eq. (4). Otherwise eq. (5).
        M_C : int
            Number of negative j channels if j_min is not directly specified.
        sample_rate : float
            Used if d is chosen automatically (d=dt).
        """
        self.data = np.asarray(data)
        self.wavelet = wavelet
        self.xi_1 = xi_1
        self.use_compensation = use_compensation
        self.sample_rate = sample_rate
        self.M_C = M_C

        # Auto-choose if needed
        if any(par is None for par in [d, b, q, j_min, j_max, alpha_for_delta]):
            auto = auto_choose_grid_params(
                data_length=len(data),
                sample_rate=sample_rate,
                M_C=M_C
            )
            d_auto, b_auto, q_auto, jmin_auto, jmax_auto, alpha_auto = auto

            self.d = d if d is not None else d_auto
            self.b = b if b is not None else b_auto
            self.q = q if q is not None else q_auto
            self.j_min = j_min if j_min is not None else jmin_auto
            self.j_max = j_max if j_max is not None else jmax_auto
            self.alpha_for_delta = alpha_for_delta if alpha_for_delta is not None else alpha_auto
        else:
            self.d = d
            self.b = b
            self.q = q
            self.j_min = j_min
            self.j_max = j_max
            self.alpha_for_delta = alpha_for_delta

        self._W = None
        self._j_values = None

    @property
    def wavelet_transform(self):
        """Compute the forward transform (W, j_values) once and cache it."""
        if self._W is None:
            W, jvals = grid_based_wavelet_transform(
                data=self.data,
                wavelet=self.wavelet,
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
        """Return the array of j-values used in the transform."""
        if self._j_values is None:
            _ = self.wavelet_transform
        return self._j_values

    def inverse_transform(self, normalization=1.0):
        """
        Reconstruct (naïve) from W[l,j], summing over l,j using eq. (4) or eq. (5).

        Parameters
        ----------
        normalization : float
            Overall amplitude scale factor.

        Returns
        -------
        data_approx : 1D ndarray (real)
        """
        if self._W is None:
            _ = self.wavelet_transform  # force compute
        data_approx = inverse_grid_based_wavelet_transform(
            W=self._W,
            j_values=self._j_values,
            wavelet=self.wavelet,
            data_length=len(self.data),
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
