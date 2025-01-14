import numpy as np
import scipy.signal

from .wavelets import Morlet

__all__ = [
    "auto_choose_grid_params",
    "FilterBankWaveletTransform",
]

def auto_choose_scales(data_length, dt, s0=None, dj=0.25):
    """
    Return a set of scales s_j = s0 * 2^(j * dj), for j=0..J.

    J = (1/dj) * log2( (N*dt) / s0 )

    data_length : int
    dt : float
    s0 : float or None, optional
    dj : float
         scale resolution in log2 space
    """
    N = data_length
    if s0 is None:
        s0 = 2 * dt
    J = int(np.ceil((1.0 / dj) * np.log2((N * dt) / s0)))
    scales = s0 * 2.0 ** (dj * np.arange(0, J + 1))
    return scales

def auto_choose_grid_params_dyadic_scales(
    data_length,
    sample_rate=1.0,
    M_C=0,
    s0=None,
    dj=0.25
):
    """
    Pick dyadic scales s_j, then solve eq(3) => j = q*(1/s_j - 1/b).

    The final j_min, j_max, etc. are deduce from that set, plus negative j from M_C.
    """
    dt = 1.0 / sample_rate
    d = dt      # decimation factor in seconds
    b = 2.0
    q = 1.0

    scales = auto_choose_scales(data_length, dt, s0=s0, dj=dj)
    j_vals = []

    for s_j in scales:
        alpha_j = (1.0 / s_j)
        j_float = q * (alpha_j - (1.0 / b))
        j_int = int(round(j_float))
        if j_int not in j_vals:
            j_vals.append(j_int)

    # negative coverage
    for neg_j in range(-M_C, 0):
        if neg_j not in j_vals:
            j_vals.append(neg_j)

    j_vals = np.array(sorted(j_vals))
    alpha_for_delta = 1.61803

    j_min = j_vals.min()
    j_max = j_vals.max() + 1
    return d, b, q, j_min, j_max, alpha_for_delta, j_vals

def auto_choose_grid_params(data_length, sample_rate=1.0, M_C=0,
                            s0=None, dj=0.25):
    """
    Incorporate the dyadic-scale approach to build j-values,
    then return (d,b,q,j_min,j_max,alpha_for_delta,j_vals).
    """
    d, b, q, j_min, j_max, alpha_for_delta, j_vals = \
        auto_choose_grid_params_dyadic_scales(
            data_length=data_length,
            sample_rate=sample_rate,
            M_C=M_C,
            s0=s0,
            dj=dj
        )
    return d, b, q, j_min, j_max, alpha_for_delta, j_vals

def _build_analysis_filter(
    wavelet,
    j_val,
    d,
    b,
    q,
    alpha,
    xi_1,
    use_compensation,
    sample_rate,
    kernel_size
):
    """
    Build the 'analysis' filter h_j for channel j_val,
    skipping if alpha_j <=0.
    """
    alpha_j = (1.0 / b) + (j_val / q)
    if alpha_j <= 0:
        return None

    dt = 1.0 / sample_rate
    half = kernel_size // 2
    t = (np.arange(kernel_size) - half) * dt
    delta_j = np.mod(alpha * j_val, 1.0)

    if use_compensation:
        wav = wavelet.grid_time_eq4(t, 0, j_val, d, b, q, delta_j)
    else:
        wav = wavelet.grid_time_eq5(t, 0, j_val, d, b, q, delta_j, xi_1)

    # analysis filter => conj(reversed(wav))
    return np.conjugate(wav[::-1])

def _build_all_analysis_filters(
    data_length,
    wavelet,
    sample_rate,
    d,
    b,
    q,
    j_min,
    j_max,
    alpha_for_delta,
    xi_1,
    use_compensation,
    kernel_size
):
    j_vals = np.arange(j_min, j_max)
    filters = {}
    valid_js = []
    for j_val in j_vals:
        h_j = _build_analysis_filter(
            wavelet, j_val, d, b, q,
            alpha_for_delta, xi_1,
            use_compensation, sample_rate,
            kernel_size
        )
        if h_j is not None:
            filters[j_val] = h_j
            valid_js.append(j_val)

    return filters, np.array(valid_js)

def _build_gram_matrix(analysis_filters, j_values):
    """
    Multi-channel Gram matrix:
       G[i,k] = <h_{j_i}, h_{j_k}> = sum_n h_{j_i}[n]* conj(h_{j_k}[n])
    """
    num_j = len(j_values)
    G = np.zeros((num_j, num_j), dtype=complex)
    for i, j1 in enumerate(j_values):
        h1 = analysis_filters[j1]
        for k, j2 in enumerate(j_values):
            h2 = analysis_filters[j2]
            G[i, k] = np.vdot(h1, h2)
    return G

def _build_multi_channel_dual(analysis_filters, j_values):
    """
    Full multi-channel dual:
      G^-1_{i,k}, then g_{j_i} = sum_k [ G^-1_{i,k} * conj(h_{j_k}) ]
    """
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

def build_synthesis_filters(
    analysis_filters,
    j_values,
    use_multi_channel=True
):
    """
    If use_multi_channel => invert Gram matrix across channels.
    If not => naive per-channel dual.
    """
    if use_multi_channel:
        return _build_multi_channel_dual(analysis_filters, j_values)
    else:
        dual_filters = {}
        for j_val, h_j in analysis_filters.items():
            norm_h = np.vdot(h_j, h_j)
            if abs(norm_h) < 1e-14:
                dual_filters[j_val] = np.zeros_like(h_j)
            else:
                dual_filters[j_val] = np.conjugate(h_j) / norm_h
        return dual_filters

class FilterBankWaveletTransform:
    """
    Filter-bank style wavelet transform with multi-channel dual,
    eq(4)/(5), negative j, dt corrections, optional zero-padding,
    and *dyadic scale* logic from auto_choose_grid_params(...).
    """

    def __init__(
        self,
        data,
        wavelet=Morlet(),
        sample_rate=1.0,
        d=None, b=None, q=None,
        j_min=None, j_max=None,
        alpha_for_delta=None,
        xi_1=0.25,
        use_compensation=False,
        M_C=0,
        use_multi_channel_dual=True,
        kernel_size=256,
        zero_pad=0,
        s0=None,
        dj=0.25
    ):
        """
        data : 1D array
        wavelet : wavelet with eq4/eq5
        sample_rate : float => dt=1/sample_rate
        d,b,q : float => grid params
        j_min,j_max => range of j
        alpha_for_delta => Kronecker multiplier
        xi_1 => eq(5)
        use_compensation => eq(4) if True, eq(5) if False
        M_C => negative j coverage
        use_multi_channel_dual => if True, invert Gram matrix for multi-channel dual
        kernel_size => length of wavelet filter
        zero_pad => integer # of samples to pad on each side
        s0,dj => for dyadic scales if the user doesn't manually set (d,b,q,j_min,j_max)
        """
        # Possibly auto-choose j-values from a dyadic scale set
        # if user hasn't specified j_min/j_max or d,b,q,alpha_for_delta
        self.data_original = np.asarray(data, dtype=float)
        self.zero_pad = zero_pad
        if zero_pad > 0:
            self.data = np.pad(self.data_original, (zero_pad, zero_pad), mode='constant')
        else:
            self.data = self.data_original

        self.wavelet = wavelet
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.xi_1 = xi_1
        self.use_compensation = use_compensation
        self.use_multi_channel_dual = use_multi_channel_dual
        self.kernel_size = kernel_size
        self.s0 = s0
        self.dj = dj

        N = len(self.data)

        # If user didn't pass in all required grid params, pick them with dyadic logic
        if any(x is None for x in [d, b, q, j_min, j_max, alpha_for_delta]):
            d_a, b_a, q_a, jmin_a, jmax_a, alpha_a, j_vals_a = auto_choose_grid_params(
                data_length=N,
                sample_rate=sample_rate,
                M_C=M_C,
                s0=self.s0,
                dj=self.dj
            )
            self.d = d if d is not None else d_a
            self.b = b if b is not None else b_a
            self.q = q if q is not None else q_a
            self.j_min = j_min if j_min is not None else jmin_a
            self.j_max = j_max if j_max is not None else jmax_a
            self.alpha_for_delta = alpha_for_delta if alpha_for_delta is not None else alpha_a
        else:
            # user-specified
            self.d = d
            self.b = b
            self.q = q
            self.j_min = j_min
            self.j_max = j_max
            self.alpha_for_delta = alpha_for_delta

        # Build analysis filters
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
            xi_1=self.xi_1,
            use_compensation=self.use_compensation,
            kernel_size=self.kernel_size
        )

        # Build dual / synthesis filters
        self.synthesis_filters = build_synthesis_filters(
            self.analysis_filters,
            self.j_values,
            use_multi_channel=self.use_multi_channel_dual
        )

    def forward_transform(self):
        """
        For each j, convolve data with the analysis filter => shape (N, num_j).
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
        data_approx = sum_j [1/sqrt(dt) * conv(W[:,j], g_j)].
        If zero_pad>0, remove the padding at the end.
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
        """
        Convenience: forward then inverse.
        """
        W = self.forward_transform()
        data_approx = self.inverse_transform(W)
        return W, data_approx
