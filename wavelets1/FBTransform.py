import numpy as np
import scipy.signal

__all__ = [
    "FilterBankWaveletTransform",
    "Morlet",
]

class Cauchy:
    def __init__(self, alpha=300):
        """
        alpha : float
            Parameter controlling the time-frequency concentration.
            Higher alpha leads to higher Q-factor.
        """
        self.alpha = alpha

    def __call__(self, t, s=1.0):
        return self.time(t, s=s)

    def time(self, t, s=1.0):
        """
        Time-domain Cauchy wavelet, centered at zero.

        t is in seconds; s is dimensionless scale.
        """
        x = t / s
        return (x + 1j)**(-(self.alpha + 1))

    def grid_time_eq4(self, t, l, j, d, b, q, delta_j):
        """
        Eq. (4):

        psi_{l,j}(t) = sqrt(1/b + j/q)* psi( (1/b + j/q)*(t - d(l+δ_j)) ).
        """
        alpha_j = (1.0 / b) + (j / q)
        shift = d * (l + delta_j)
        arg = alpha_j * (t - shift)
        return np.sqrt(alpha_j) * self.time(arg, s=1.0)

    def grid_time_eq5(self, t, l, j, d, b, q, delta_j, xi_1):
        """
        Eq. (5):

        psi_{l,j}^comp(t) = (1/sqrt(b)) * psi((t - d(l+δ_j))/b)
            * exp(2π i xi_1 * j*(t - d(l+δ_j)) / q).
        """
        shift = d * (l + delta_j)
        tau = (t - shift) / b
        phase = np.exp(2j * np.pi * xi_1 * j * (t - shift) / q)
        return (1.0 / np.sqrt(b)) * self.time(tau, s=1.0) * phase

###############################################################################
# (1) Morlet wavelet with eq(4) and eq(5). 
#     We'll store xi_1 internally, but eq(4) doesn't use it.
###############################################################################
class Morlet:
    """
    Complex Morlet wavelet that implements both eq(4) and eq(5),
    deciding which to use based on alpha_j > 0 or not.
    """

    def __init__(self, w0=6, xi_1=0.25):
        """
        w0  : float, nondimensional frequency constant (~6 recommended).
        xi_1: float, center frequency factor used in eq(5).
        """
        self.w0 = w0
        self.xi_1 = xi_1

    def __call__(self, t, s=1.0, complete=True):
        return self.time(t, s=s, complete=complete)

    def time(self, t, s=1.0, complete=False):
        """
        Standard Morlet time-domain wavelet for dimensionless scale s,
        with center frequency w0. 
        t is in seconds if dt is handled externally.
        """
        w = self.w0
        x = t / s
        out = np.exp(1j * w * x)
        if complete:
            out -= np.exp(-0.5 * (w ** 2))
        out *= np.exp(-0.5 * (x ** 2)) * np.pi ** (-0.25)
        return out

    def eq4_main_scale(self, t, l, j, d, b, q, delta_j):
        """
        Eq.(4) for alpha_j > 0:
          psi_{l,j}(t) = sqrt(1/b + j/q)* psi( [1/b + j/q]*( t - d(l+δ_j)) ).
        """
        alpha_j = (1.0 / b) + (j / q)
        shift = d * (l + delta_j)   
        arg = alpha_j * (t - shift) 
        base = self.time(arg, s=1.0, complete=True)
        return np.sqrt(alpha_j) * base

    def eq5_compensation(self, t, l, j, d, b, q, delta_j):
        """
        Eq.(5) for alpha_j <= 0 => "compensation" wavelet:
          psi_{l,j}^comp(t) = (1/sqrt(b)) * psi((t - d(l+δ_j))/b)
                              * exp(2π i xi_1 * j*(t - d(l+δ_j))/q).
        """
        shift = d * (l + delta_j)
        tau = (t - shift) / b
        base = self.time(tau, s=1.0, complete=False)
        phase = np.exp(
            2j * np.pi * self.xi_1 * j * (t - shift) / q
        )
        return (1.0 / np.sqrt(b)) * base * phase

###############################################################################
# (2) Build analysis filters with piecewise logic for eq(4)/(5)
###############################################################################
def _build_analysis_filter(
    wavelet,
    j_val,
    d, b, q,
    alpha,
    sample_rate,
    kernel_size,
    l_val=0
):
    """
    For channel j_val, we do:
      - eq(4) if alpha_j > 0
      - eq(5) if alpha_j <= 0
    and then conj(reversed) to build the analysis filter.
    """
    dt = 1.0 / sample_rate
    half = kernel_size // 2
    t = (np.arange(kernel_size) - half) * dt

    delta_j = np.mod(alpha * j_val, 1.0)
    alpha_j = (1.0 / b) + (j_val / q)

    # Decide eq(4) vs eq(5)
    if alpha_j > 0:
        wav = wavelet.eq4_main_scale(t, l_val, j_val, d, b, q, delta_j)
    else:
        wav = wavelet.eq5_compensation(t, l_val, j_val, d, b, q, delta_j)

    # analysis => conj(reversed(wav))
    return np.conjugate(wav[::-1])

def _build_all_analysis_filters(
    data_length,
    wavelet,
    sample_rate,
    d, b, q,
    j_min, j_max,
    alpha_for_delta,
    kernel_size
):
    j_vals = np.arange(j_min, j_max+1)
    filters = {}
    valid_js = []

    for j_val in j_vals:
        h_j = _build_analysis_filter(
            wavelet=wavelet,
            j_val=j_val,
            d=d, b=b, q=q,
            alpha=alpha_for_delta,
            sample_rate=sample_rate,
            kernel_size=kernel_size
        )
        filters[j_val] = h_j/data_length
        valid_js.append(j_val)

    return filters, np.array(valid_js)

###############################################################################
# (3) Synthesis filters => multi-channel dual or naive
###############################################################################
def _build_gram_matrix(analysis_filters, j_values):
    num_j = len(j_values)
    G = np.zeros((num_j, num_j), dtype=complex)
    for i, j1 in enumerate(j_values):
        h1 = analysis_filters[j1]
        for k, j2 in enumerate(j_values):
            h2 = analysis_filters[j2]
            G[i, k] = np.vdot(h1, h2)  # sum conj(h1)*h2
    return G

def _build_multi_channel_dual(analysis_filters, j_values):
    G = _build_gram_matrix(analysis_filters, j_values)
    G_inv = np.linalg.inv(G)
    dual_filters = {}
    for i, j1 in enumerate(j_values):
        length = len(analysis_filters[j1])
        g1 = np.zeros(length, dtype=complex)
        for k, j2 in enumerate(j_values):
            conj_h2 = np.conjugate(analysis_filters[j2])
            factor = G_inv[i, k]
            g1 += factor * conj_h2
        dual_filters[j1] = g1
    return dual_filters

def build_synthesis_filters(analysis_filters, j_values, use_multi_channel=True, dt=1.0):
    """
    If multi_channel => invert Gram => multi-channel dual
    Else => naive => g_j = conj(h_j)/||h_j||^2
    Then multiply by sqrt(dt) so the inverse transform can multiply partial sums
    by 1/sqrt(dt).
    """
    scale_factor = np.sqrt(dt)
    if use_multi_channel:
        dual_filters = _build_multi_channel_dual(analysis_filters, j_values)
        for j_val in dual_filters:
            dual_filters[j_val] *= scale_factor
        return dual_filters
    else:
        dual_filters = {}
        for j_val, h_j in analysis_filters.items():
            norm_h = np.vdot(h_j, h_j)
            if abs(norm_h) < 1e-14:
                dual_filters[j_val] = np.zeros_like(h_j)
            else:
                dual_filters[j_val] = (np.conjugate(h_j) / norm_h) * scale_factor
        return dual_filters

###############################################################################
# (4) FilterBankWaveletTransform => eq(4)/(5) piecewise, dt usage consistent
###############################################################################
class FilterBankWaveletTransform:
    """
    Filter-bank wavelet transform that uses eq(4) if alpha_j>0, eq(5) if alpha_j<=0,
    ensuring we have "main scale" for positive j and "compensation" for negative j,
    as the paper indicates. We do:

      alpha_j = 1/b + j/q

      if alpha_j > 0 => eq(4)
      else           => eq(5)

    We multiply by sqrt(dt) in the forward transform, 1/sqrt(dt) in the inverse,
    keep wavelet's center freq internally if eq(5) is used, etc.
    """

    def __init__(
        self,
        data,
        wavelet=Morlet(w0=6, xi_1=0.25),
        sample_rate=1.0,
        d=1.0, b=2.0, q=1.0,
        j_min=-1, j_max=16,
        alpha_for_delta=1.61803,
        use_multi_channel_dual=True,
        kernel_size=256,
        zero_pad=0,
    ):
        """
        data : 1D array
        wavelet: wavelet class implementing eq(4)/(5) piecewise.
        sample_rate => dt=1/sample_rate
        d,b,q => eq(3) grid parameters
        j_min,j_max => freq channel range
        alpha_for_delta => quasi-random offset
        use_multi_channel_dual => invert Gram matrix for channels
        kernel_size => wavelet filter length in samples
        zero_pad => # of samples to pad
        """
        self.data_original = np.asarray(data, dtype=float)
        self.zero_pad = zero_pad
        if zero_pad > 0:
            self.data = np.pad(self.data_original, (zero_pad, zero_pad), mode='constant')
        else:
            self.data = self.data_original

        self.wavelet = wavelet
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.use_multi_channel_dual = use_multi_channel_dual
        self.kernel_size = kernel_size

        self.d = d   # in seconds
        self.b = b   # dimensionless
        self.q = q   # dimensionless
        self.j_min = j_min
        self.j_max = j_max
        self.alpha_for_delta = alpha_for_delta

        N = len(self.data)

        # Build piecewise eq(4)/(5) analysis filters
        self.analysis_filters, self.j_values = _build_all_analysis_filters(
            data_length=N,
            wavelet=self.wavelet,
            sample_rate=self.sample_rate,
            d=self.d,
            b=self.b,
            q=self.q,
            j_min=self.j_min,
            j_max=self.j_max,
            alpha_for_delta=self.alpha_for_delta,
            kernel_size=self.kernel_size
        )

        # Build dual / synthesis filters
        self.synthesis_filters = build_synthesis_filters(
            self.analysis_filters,
            self.j_values,
            use_multi_channel=self.use_multi_channel_dual,
            dt=self.dt
        )

    def forward_transform(self):
        """
        For each j, convolve data with analysis filter => shape (N, num_j).
        Multiply by sqrt(dt).
        """
        N = len(self.data)
        num_j = len(self.j_values)
        W = np.zeros((N, num_j), dtype=complex)

        scale_forward = np.sqrt(self.dt)

        for idx, j_val in enumerate(self.j_values):
            h_j = self.analysis_filters[j_val]
            conv_out = scipy.signal.fftconvolve(self.data, h_j, mode='same')
            W[:, idx] = scale_forward * conv_out

        return W

    def inverse_transform(self, W):
        """
        data_approx = sum_j [ (1/sqrt(dt)) * conv(W[:,j], g_j ) ].
        Remove padding if zero_pad>0
        """
        N = len(self.data)
        data_rec = np.zeros(N, dtype=complex)
        scale_inverse = 1.0 / np.sqrt(self.dt)

        for idx, j_val in enumerate(self.j_values):
            g_j = self.synthesis_filters[j_val]
            c_out = scipy.signal.fftconvolve(W[:, idx], g_j, mode='same')
            data_rec += scale_inverse * c_out

        if self.zero_pad > 0:
            data_rec = data_rec[self.zero_pad : -self.zero_pad]

        return data_rec.real

    def run_full_transform(self):
        """Convenience method => forward + inverse."""
        W = self.forward_transform()
        data_approx = self.inverse_transform(W)
        return W, data_approx
