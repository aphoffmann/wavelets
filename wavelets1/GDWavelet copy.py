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
import scipy.signal
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.signal import fftconvolve

class Cauchy:
    def __init__(self, alpha=300, epsilon=1e-2):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        # Compute normalization factor using the provided formula
        term1 = -2 * np.log(2 * np.pi)
        term2 = -0.5 * gammaln(2 * self.alpha + 1)
        term3 = gammaln(self.alpha + 1)
        term4 = ((2 * self.alpha + 2) / 2) * np.log(2)
        term5 = np.log(self.alpha)
        self.norm = np.exp(term1 + term2 + term3 + term4 + term5)

    def eval_analysis(self, t):
        """Evaluate the analysis wavelet function at time t using NumPy for complex arithmetic."""
        t = np.asarray(t, dtype=np.float64)  # Ensure t is a NumPy array or float
        factor = 1 - 2j * np.pi * t / self.alpha
        analysis_wavelet = np.power(factor, -1 - self.alpha)
        return analysis_wavelet

    def eval_synthesis(self, t):
        """Evaluate the synthesis wavelet function at time t."""
        return self.norm * self.eval_analysis(t).conj()
    
    def eq3(self, t, j, b, q,):
        """
        Compute Equation 4: ψ_{l,j}(t) for α_j > 0.
        
        Equation 4:
        ψ_{l,j}(t) = sqrt(1/b + j/q) * ψ((1/b + j/q) * (t - d(l + δ_j))).
        """
        alpha_j = (1.0 / b) + (j / q)
        arg = alpha_j * t
        return np.sqrt(alpha_j) * self.eval_analysis(arg)
    
    def eq4(self, t, j, b, q, xi_1):
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
    
    def eq3_synth(self, t, l, j, d, b, q, delta_j):
        """
        Compute the synthesis counterpart of Equation 4 for ψ_{l,j}(t) using synthesis wavelet.
        Adjusted to use eval_synthesis for reconstruction.
        """
        alpha_j = (1.0 / b) + (j / q)
        shift = d * (l+ delta_j)
        arg = alpha_j * (t - shift)
        # Use synthesis wavelet function with appropriate scaling. 
        return np.sqrt(alpha_j) * self.eval_synthesis(arg)

    def eq4_synth(self, t, l, j, d, b, q, delta_j, xi_1):
        """
        Compute the synthesis counterpart of Equation 5 for ψ_{l,j}^{comp}(t) using synthesis wavelet.
        """
        shift = d * (l + delta_j) 
        tau = (t - shift) / b
        base = self.eval_synthesis(tau)
        phase = np.exp(2j * np.pi * xi_1 * j * (t - shift) / q)
        return (1.0 / np.sqrt(b)) * base * phase

    def eval_repkern(self, a, b):
        """Evaluate the reproducing kernel at scale a and position b."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        c = (self.alpha * np.log(a)
             + gammaln(2 * self.alpha)
             - (1 + 2 * self.alpha) * np.log(2 * np.pi))
        factor = 1 + a - 2j * np.pi * b / self.alpha
        return np.exp(c) * np.power(factor, -1 - 2 * self.alpha)

    def cutoff_time(self):
        """Estimate cutoff time based on epsilon."""
        return self.alpha * np.sqrt(self.epsilon ** (-2/(self.alpha+1)) - 1) / (2 * np.pi)

    def cutoff_freq(self):
        """Estimate cutoff frequency based on epsilon."""
        denominator = (self.alpha ** 2) * (self.epsilon ** (-2/(self.alpha+1)) - 1) / ((2*np.pi) ** 2)
        return 1 + 1/denominator

class GridWaveletTransform:
    def __init__(self, data, fs, d=1, b=None, q = None, M=None, Mc=None, xi_1 = None, wavelet=Cauchy()):
        self.data = np.asarray(data, dtype=float)

        self.fs = fs             # Sampling frequency
        self.d = d               # Decimation factor (Uniform Decimation)
        self.b = b               # largest scale factor
        self.q = q               # frequency step factor NOTE: Larger Q is better
        self.wavelet = wavelet   # Cauchy wavelet
        self.M = M               # Number of wavelet scales
        self.Mc = Mc             # Number of compensation scales for lower frequencies
        self.xi_1 = xi_1         # Lowpass Filter for compensation channels
        self.norm = None

        # Calculate optimal parameters if None
        self.calcluate_parameters()

        # Calculate time vector, translations, and scale channels
        self.time = np.arange(0, self.data.shape[-1] / self.fs, 1 / self.fs)
        self.l_channels = np.arange(self.data.shape[-1] // self.d)
        self.j_channels = np.arange(self.Mc + self.M)

        # Calculate quasi-random scale-channel delays
        self.delta_js = self.delay(np.arange(-self.Mc, self.M))

    def calcluate_parameters(self):
        # Calculate optimal parameters if None
        if self.b is None:
            self.b = (self.data.shape[-1] / 10 / self.fs)

        if self.q is None: # todo calculate q and B based on window size
            self.q = self.b

        if self.M is None:
            # 8 channels times the number of octaves from lowest frequency to Nyquist
            self.M = int(self.q*(self.fs/2 - 1/self.b))

        if self.Mc is None:
            self.Mc = 1

        if self.xi_1 is None:
            self.xi_1 = (self.fs * self.q) / self.Mc / 2
        
        return

    def forward(self):  
        coeffs = np.zeros((self.j_channels.shape[0], self.data.shape[0]), dtype=complex)
        time = self.time - np.mean(self.time)
        for idx, j in enumerate(self.j_channels):
            # Note: Higher j corresponds to higher frequency
            # b is the largest scale of interest
            # q is the frequency step factor
            delta_j = self.delta_js[idx]

            if(j >= self.Mc):
                # Eq 3
                filterbank = self.wavelet.eq3(time, j, self.b, self.q)
            else:
                # Eq 4
                filterbank = self.wavelet.eq4(time, j,  self.b, self.q, self.xi_1)

            coeffs[j] = np.array(fftconvolve(self.data, filterbank, mode='same')) 
            
        return(coeffs)

    def inverse(self, coeffs):
        # Coeffs (j, l)
        time = self.time - np.mean(self.time)
        result = np.zeros(self.data.shape, dtype=complex)

        if(self.norm is None):
            self.norm = np.sum(np.abs(self.wavelet.eq3(time, self.j_channels[-1],  self.b, self.q))**2)
            

        for idx, j in enumerate(self.j_channels):
            delta_j = self.delta_js[idx]
            if(j >= self.Mc):
                # Eq 3
                filterbank = self.wavelet.eq3(time, j, self.b, self.q)
                sj = 1 / (1/self.b + j / self.q)
            else:
                # Eq 4
                filterbank = self.wavelet.eq4(time, j,  self.b, self.q, self.xi_1)
                sj = 1 / self.b

            result += np.array(fftconvolve(coeffs[j], filterbank, mode='same')) / self.q * np.sqrt( 1/self.fs) / self.norm

        
        return result.real

    def delay(self, channels):
        """Calculate the delay vectors for each channel."""
        alpha = 1 - 2 / (1 + np.sqrt(5))  # 1 - 1/(golden ratio0
        return(np.mod(channels * alpha + 0.5, 1) - 0.5)
        

    def framebounds(self):
        pass

    


    
####################################### Test Plots
'''
alpha = 30                   # Example alpha value
sample_rate = 50              # 50 Hz sample rate
duration = 10.0                # Duration in seconds for the plot
t = np.arange(-duration, duration, 1/sample_rate)

# Instantiate the wavelet
wavelet = CauchyWavelet(alpha=alpha)

# Evaluate the wavelet
wavelet_values = wavelet.eval_analysis(t)

# Extract real and imaginary parts for plotting
real_part = np.real(wavelet_values)
imag_part = np.imag(wavelet_values)
magnitude = np.abs(wavelet_values)
phase = np.angle(wavelet_values)

# Plotting the wavelet
plt.figure(figsize=(12, 8))

# Real and Imaginary parts
plt.subplot(2, 1, 1)
plt.plot(t, real_part, label='Real part')
plt.plot(t, imag_part, label='Imaginary part', linestyle='--')
plt.title('Cauchy Wavelet (α = {})'.format(alpha))
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Magnitude and Phase
plt.subplot(2, 1, 2)
plt.plot(t, magnitude, label='Magnitude')
plt.plot(t, phase, label='Phase', linestyle='--')
plt.title('Magnitude and Phase')
plt.xlabel('Time [s]')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

alpha = 900                   # Example alpha value
sample_rate = 1              # 50 Hz sample rate


duration = 1000.0                # Duration in seconds for the plot
t = np.arange(0, duration, 1/sample_rate)
N = duration*sample_rate


# Instantiate the wavelet
wavelet = Cauchy(alpha=alpha)


# grid decimated
l_channels = np.arange(N//sample_rate)
fs = 50; l = l_channels[5]; j = 0;


d = 50; b = N//(sample_rate*5)/wavelet.cutoff_time(); q = 1;
delta_j = 0

# Evaluate the wavelet
wavelet_values = wavelet.eq3(t, fs, l, j, d, b, q, delta_j)


# Extract real and imaginary parts for plotting
real_part = np.real(wavelet_values)
imag_part = np.imag(wavelet_values)
magnitude = np.abs(wavelet_values)
phase = np.angle(wavelet_values)

# Plotting the wavelet
plt.figure(figsize=(12, 8))

# Real and Imaginary parts
plt.subplot(2, 1, 1)
plt.plot(t, real_part, label='Real part')
plt.plot(t, imag_part, label='Imaginary part', linestyle='--')
plt.title('Cauchy Wavelet (α = {})'.format(alpha))
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot fft of wavelet
fft_values = np.fft.fft(wavelet_values)
fft_freq = np.fft.fftfreq(len(fft_values), d=1/sample_rate)

# Only take the positive half of frequencies for plotting
half_n = len(fft_values) // 2
fft_values_positive = fft_values[:half_n]
fft_freq_positive = fft_freq[:half_n]
fft_magnitude = (2.0 / N) * np.abs(fft_values_positive)
fft_magnitude[0] = fft_magnitude[0] / 2

plt.subplot(2, 1, 2)
plt.plot(fft_freq_positive, fft_magnitude, label='Magnitude')
plt.xscale('log')


plt.title('FFT of Cauchy Wavelet (α = {})'.format(alpha))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')

plt.grid(True)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("B: ", b)
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