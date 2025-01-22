"""
grid_wavelet.py

A module for implementing the Grid-Based Decimated Wavelet Transform
with Stably Invertible Implementation based on the paper:
"Grid-Based Decimation for Wavelet Transforms With Stably Invertible Implementation."

This module outlines the key classes and functions needed to construct
analysis and synthesis filter banks, perform forward and inverse transforms,
and utilize quasi-random delays via low-discrepancy sequences.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy import signal

__all__ = ["Morlet", "Cauchy", "GridWaveletTransform"]

class Morlet:
    def __init__(self, w0=6):
        """w0 is the nondimensional frequency constant. If this is
        set too low then the wavelet does not sample very well: a
        value over 5 should be ok; Terrence and Compo set it to 6.
        """
        self.w0 = w0
        if w0 == 6:
            # value of C_d from TC98
            self.C_d = 0.776

    def eval_analysis(self, t):
        w = self.w0

        x = t
        output = np.exp(1j * w * x)
        output -= np.exp(-0.5 * (w ** 2))
        output *= np.exp(-0.5 * (x ** 2)) * np.pi ** (-0.25)

        return output
    
    def eq3_analysis(self, t, j, b, q,):
        """
        Compute Equation 4: ψ_{l,j}(t) for α_j > 0.
        
        Equation 4:
        ψ_{l,j}(t) = sqrt(1/b + j/q) * ψ((1/b + j/q) * (t - d(l + δ_j))).
        """
        alpha_j = (1.0 / b) + (j / q)
        arg = alpha_j * t
        return np.sqrt(alpha_j) * self.eval_analysis(arg)
    
    def eq4_analysis(self, t, j, b, q, xi_1):
        """
        Compute Equation 5: ψ_{l,j}^{comp}(t) for α_j ≤ 0.
        
        Equation 5:
        ψ_{l,j}^{comp}(t) = (1/√b) ψ((t - d(l+δ_j)) / b)
                          * exp(2πi ξ₁ * j (t - d(l+δ_j)) / q).
        """
        tau = t / b 
        base = self.eval_analysis(tau) # TODO
        phase = np.exp(2j * np.pi * xi_1 * j * (t) / q)
        return (1.0 / np.sqrt(b)) * base * phase

class Cauchy:
    def __init__(self, alpha=300, epsilon=1e-2):
        """
        Defines the "analysis" wavelet family. We omit the 'synthesis' methods
        since we will perform inversion by the frame-operator formula.
        """
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        # Normalization factor used if we wanted a direct "eval_synthesis"
        term1 = -2 * np.log(2 * np.pi)
        term2 = -0.5 * gammaln(2 * self.alpha + 1)
        term3 = gammaln(self.alpha + 1)
        term4 = ((2 * self.alpha + 2) / 2) * np.log(2)
        term5 = np.log(self.alpha)
        self.norm = np.exp(term1 + term2 + term3 + term4 + term5)

    def eval_analysis(self, t):
        """Continuous-time 'analysis' mother wavelet."""
        # factor^(-1 - alpha)
        factor = 1 - 2j * np.pi * t / self.alpha
        return factor ** (-1 - self.alpha)

    def eq3_analysis(self, t, j, b, q):
        """
        eq3 wavelet, alpha_j > 0
        psi_j(t) = sqrt(alpha_j) * mother(alpha_j * t).
        """
        alpha_j = (1.0 / b) + (j / q)
        return np.sqrt(alpha_j) * self.eval_analysis(alpha_j * t)

    def eq4_analysis(self, t, j, b, q, xi_1):
        """
        eq4 wavelet, alpha_j <= 0
        psi_j(t) = (1/sqrt(b)) * mother(t/b) * exp(+2 pi i xi_1 j t / q).
        """
        tau = t / b
        base = self.eval_analysis(tau)
        phase = np.exp(+2j * np.pi * xi_1 * j * t / q)
        return (1.0 / np.sqrt(b)) * base * phase

    # ----------------------------------------------------------------
    # Optional helpers for time/freq cutoff estimates:
    # ----------------------------------------------------------------
    def cutoff_time(self):
        """
        Estimate a time cutoff for wavelet support (where amplitude < epsilon).
        """
        return self.alpha * np.sqrt(self.epsilon ** (-2/(self.alpha+1)) - 1) / (2*np.pi)

    def cutoff_freq(self):
        """
        Estimate a frequency cutoff for wavelet support (where amplitude < epsilon).
        """
        denominator = ((self.alpha ** 2) *
                       (self.epsilon ** (-2/(self.alpha+1)) - 1) /
                       ((2*np.pi) ** 2))
        return 1 + 1/denominator

class GridWaveletTransform:
    """
    Implements a non-decimated wavelet transform:

    Forward transform: 
       c2D[j, :] = ifft( fft(data) * fft(wavelet_j) )

    Inverse transform via "frame operator" S(w):
       X_hat(w) = (1 / S(w)) * sum_j conj(W_j(w)) * fft( c2D[j, :] )
       x_hat(n) = ifft( X_hat(w) )

    Where S(w) = sum_j |W_j(w)|^2,  the sum of wavelet magnitude-squares across channels.
    """

    def __init__(self, data, fs, wavelet=Cauchy(),
                b=None, q = None, M=None, Mc=None, xi_1 = None,
                pad_method='symmetric'):
        
        
        self.data = np.asarray(data, dtype=float)
        self.N = self.data.shape[-1] # Number of samples
        self.fs = fs             # Sampling frequency
        self.wavelet = wavelet   # Cauchy wavelet


        # Pad data
        self.pad_width = 0
        if(pad_method is not None):
            self.pad_width = data.shape[-1]
            self.data = np.pad(self.data, self.pad_width, mode=pad_method)
            self.N = self.data.shape[-1]
            self.data *= signal.windows.tukey(self.N, alpha=0.3)

         # fill default parameters
        self._init_params(b, q, M, Mc, xi_1)

        # create time vector, centered so wavelet is around t=0
        self.time = np.arange(self.N) / self.fs
        self.time -= np.mean(self.time)

        # define wavelet channels
        self.j_channels = np.arange(-self.Mc, self.M)
        self.delays = self._delay(self.j_channels)



        # Precompute the wavelets in frequency domain, Wfreq[j, :], shape = (#channels, N).
        self.Wfreq = self._build_wavelets_FD()

            # Compute the phase shift for time correction  
        self.freqs = np.fft.fftfreq(self.N)
        self.phase_shift = np.exp(-1j * 2 * np.pi * self.freqs * (self.N / 2))
        self.Wfreq *= self.phase_shift[np.newaxis, :]

        # Frame operator S(w) = sum_j |W_j(w)|^2
        # shape = (N,)
        self.Sfreq = np.sum(np.abs(self.Wfreq)**2, axis=0)

        # Avoid dividing by zero in case some frequency bins are extremely small:
        eps = 1e-12
        self.Sfreq[self.Sfreq < eps] = eps
    
    def _init_params(self, b, q, M, Mc, xi_1):
        self.b = b
        self.q = q
        self.M = M
        self.Mc = Mc
        self.xi_1 = xi_1

        if self.b is None:
            self.b = self.N / (2 * self.fs)
        if self.q is None:
            self.q = self.b
        if self.M is None:
            # just an example guess
            self.M = int(self.q*(self.fs/2 - 1/self.b))
            if self.M < 1:
                self.M = 4

        if self.Mc is None:
            self.Mc = int(self.q/self.b)
            if self.Mc < 1:
                self.Mc = 1

        if self.xi_1 is None:
            self.xi_1 = (self.fs * self.q) / max(self.M,1) / 2

    def _delay(self, j_channels):
        """Low-discrepancy delay (like in filter bank 4)."""
        alpha = 1 - 2/(1+np.sqrt(5))  # golden-ratio fraction
        return (np.mod(j_channels * alpha + 0.5, 1) - 0.5)
    
    def _build_wavelets_FD(self):
        """
        Build the frequency-domain wavelets W_j(w). For each channel j:
            wavelet_j(t) = eq3_analysis or eq4_analysis with shift 'delays[j]'.
            Then W_j(w) = fft( wavelet_j(t) ).
        Returns Wfreq of shape (#channels, N).
        """
        jvals = self.j_channels
        Wfreq = np.zeros((len(jvals), self.N), dtype=complex)
        for i, j in enumerate(jvals):
            shift = self.delays[i]
            if (j/self.q + 1/self.b) > 0:
                # eq3
                wtime = self.wavelet.eq3_analysis(self.time - shift, j, self.b, self.q)
            else:
                # eq4
                wtime = self.wavelet.eq4_analysis(self.time - shift, j, self.b, self.q, self.xi_1)

            Wfreq[i,:] = np.fft.fft(wtime)
        return Wfreq
    
    def forward(self):
        """
        For each channel j:
          coeffs[j, :] = ifft( fft(data) * Wfreq[j, :] )
        shape: (#channels, N)
        """
        Fdata = np.fft.fft(self.data)
        J = self.Wfreq.shape[0]
        coeffs = np.zeros((J, self.N), dtype=complex)

        for j in range(J):
            coeffs[j, :] = np.fft.ifft(Fdata * self.Wfreq[j, :])
        return coeffs
    
    def inverse(self, coeffs):
        """
        coeffs is (#channels, N).
        1) Convert each row to frequency domain: c2Dfreq[j, :] = fft( c2D[j, :] ).
        2) Sum_j [ conj(W_j(w)) * c2Dfreq_j(w ) ] / Sfreq(w).
        3) ifft -> xhat(n).
        """
        J = coeffs.shape[0]
        c2Dfreq = np.zeros_like(coeffs, dtype=complex)  # same shape
        for j in range(J):
            c2Dfreq[j, :] = np.fft.fft(coeffs[j, :])

        # Weighted sum in freq: XhatFreq = [1/Sfreq] * sum_j conj(Wfreq[j,:]) * c2Dfreq[j,:]
        numerator = np.zeros(self.N, dtype=complex)
        for j in range(J):
            numerator += np.conjugate(self.Wfreq[j,:]) * c2Dfreq[j,:]

        XhatFreq = numerator / self.Sfreq
        xhat_time = np.fft.ifft(XhatFreq).real

        # Remove padding
        if self.pad_width > 0:
            xhat_time = xhat_time[self.pad_width:-self.pad_width]

        return xhat_time
    
    def overlap_save_conv(self, x, h, block_size=2048):
        """
        Perform linear convolution of x with h using the Overlap-Save method.

        Parameters
        ----------
        x : 1D array
            The input signal
        h : 1D array
            The filter (wavelet in your case)
        block_size : int
            The size of each processing block. Must be >= len(h).

        Returns
        -------
        y : 1D array
            The linear convolution result, length = len(x) + len(h) - 1
            (unless you choose to trim it to match x's length).
        """
        L = len(h)
        if block_size < L:
            raise ValueError("block_size must be at least as large as len(h).")

        # We'll zero-pad h to length block_size
        H = np.fft.fft(h, n=block_size)

        # The output length for linear conv is len(x)+len(h)-1
        out_len = len(x) + L - 1
        y = np.zeros(out_len, dtype=np.complex128)

        # Number of new samples we can process each block
        step_size = block_size - (L - 1)

        # We'll maintain a buffer of length block_size,
        # reading step_size new samples each time.
        x_pos = 0

        # We can keep processing until we've consumed all of x
        while x_pos < len(x):
            # Copy block_size samples into a temp array (with overlap)
            block = np.zeros(block_size, dtype=np.complex128)

            # The new portion is x[x_pos : x_pos+step_size],
            # but we also need the (L-1) overlap from the end of the last block.
            end_pos = min(x_pos + step_size, len(x))
            block_data = x[x_pos:end_pos]
            block[0:len(block_data)] = block_data

            # FFT
            X_block = np.fft.fft(block, n=block_size)
            # Multiply in freq domain
            Y_block = X_block * H
            # IFFT
            y_block = np.fft.ifft(Y_block)

            # Output starts after the first (L-1) corrupted samples
            # because overlap-save discards those
            start_out = x_pos
            end_out   = start_out + step_size
            y[start_out:end_out] += y_block[L-1 : L-1 + step_size]

            # Advance
            x_pos += step_size

        return y


####################################### Test Plots
'''
alpha = 900                   # Example alpha value
sample_rate = 50              # 50 Hz sample rate
duration = 100.0                # Duration in seconds for the plot
t = np.arange(0, duration, 1/sample_rate)
N = duration*sample_rate
data = np.sin(2*np.pi*5*t)

# Instantiate the wavelet
wavelet =Cauchy(alpha=alpha)




wt = GridWaveletTransform(data, sample_rate, wavelet=wavelet, d = 10, Mc=4)
coeffs = wt.forward()
result = wt.inverse(coeffs)

# Plotting the wavelet
plt.figure(figsize=(12, 8))

# Real and Imaginary parts
plt.subplot(2, 1, 1)
extent = [0, wt.time[-1], 0,25]
plt.imshow(np.abs(coeffs), aspect="auto", origin="lower",extent = extent)




plt.legend()
plt.grid(True)

def getFFT(result, sample_rate = sample_rate):
    # Plot fft of wavelet
    fft_values = np.fft.fft(result)
    fft_freq = np.fft.fftfreq(len(fft_values), d=1/sample_rate)

    # Only take the positive half of frequencies for plotting
    half_n = len(fft_values) // 2
    fft_values_positive = fft_values[:half_n]
    fft_freq_positive = fft_freq[:half_n]
    fft_magnitude = (2.0 / N) * np.abs(fft_values_positive)
    fft_magnitude[0] = fft_magnitude[0] / 2
    return(fft_freq_positive, fft_magnitude)

plt.subplot(2, 1, 2)
f1, reconstruction = getFFT(result)
f2, data_f = getFFT(data)
plt.plot(f1, reconstruction, label='Reconstruction')
plt.plot(f2, data_f, label = "original")
plt.yscale('log')

 
plt.title('FFT of Cauchy Wavelet (α = {})'.format(alpha))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')

plt.grid(True)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("B: ", wt.b)
print("period: ", 1/fft_freq_positive[np.argmax(fft_magnitude)])
print("Percent duration; ", (1/fft_freq_positive[np.argmax(fft_magnitude)]) / duration)


'''

"""
# TEst reconstruction
# %%
alpha = 900                   # Example alpha value
sample_rate = 50              # 50 Hz sample rate
duration = 100.0                # Duration in seconds for the plot
t = np.arange(0, duration, 1/sample_rate)
N = duration*sample_rate
data = np.sin(2*np.pi*5*t)

# Instantiate the wavelet
wavelet = Cauchy(alpha=alpha)
wt = GridWaveletTransform(data, sample_rate, wavelet=wavelet, d = 10, Mc=-4)
coeffs = wt.forward()
result = wt.inverse(coeffs)

# Plotting the wavelet
plt.figure(figsize=(12, 8))

# Real and Imaginary parts
plt.subplot(2, 1, 1)
plt.imshow(np.log(np.abs(coeffs.T)), aspect="auto", origin="upper")
plt.legend()
plt.grid(True)

def getFFT(result, sample_rate = sample_rate):
    # Plot fft of wavelet
    fft_values = np.fft.fft(result)
    fft_freq = np.fft.fftfreq(len(fft_values), d=1/sample_rate)

    # Only take the positive half of frequencies for plotting
    half_n = len(fft_values) // 2
    fft_values_positive = fft_values[:half_n]
    fft_freq_positive = fft_freq[:half_n]
    fft_magnitude = (2.0 / N) * np.abs(fft_values_positive)
    fft_magnitude[0] = fft_magnitude[0] / 2
    return(fft_freq_positive, fft_magnitude)

plt.subplot(2, 1, 2)
f1, reconstruction = getFFT(result)
f2, data_f = getFFT(data)
plt.plot(f1, reconstruction, label='Reconstruction')
plt.plot(f2, data_f, label = "original")
plt.yscale('log')


plt.title('FFT of Cauchy Wavelet (α = {})'.format(alpha))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')

plt.grid(True)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("B: ", wt.b)
print("period: ", 1/fft_freq_positive[np.argmax(fft_magnitude)])
print("Percent duration; ", (1/fft_freq_positive[np.argmax(fft_magnitude)]) / duration)

"""