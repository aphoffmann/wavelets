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
        return np.power(factor, -1 - self.alpha)

    def eval_synthesis(self, t):
        """Evaluate the synthesis wavelet function at time t."""
        return self.norm * self.eval_analysis(t)
    
    def eq3(self, t, fs, l, j, d, b, q, delta_j):
        """
        Compute Equation 4: ψ_{l,j}(t) for α_j > 0.
        
        Equation 4:
        ψ_{l,j}(t) = sqrt(1/b + j/q) * ψ((1/b + j/q) * (t - d(l + δ_j))).
        """
        alpha_j = (1.0 / b) + (j / q)
        shift = d * (l + delta_j) / fs
        arg = alpha_j * (t - shift)
        return np.sqrt(alpha_j) * self.eval_analysis(arg) # TODO

    def eq4(self, t, fs, l, j, d, b, q, delta_j, xi_1):
        """
        Compute Equation 5: ψ_{l,j}^{comp}(t) for α_j ≤ 0.
        
        Equation 5:
        ψ_{l,j}^{comp}(t) = (1/√b) ψ((t - d(l+δ_j)) / b)
                          * exp(2πi ξ₁ * j (t - d(l+δ_j)) / q).
        """
        shift = d * (l + delta_j)
        tau = (t - shift) / b / fs
        base = self.eval_analysis(tau) # TODO
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
    def __init__(self, data, fs=1, d=None, b=None, q = None, M=None, Mc=None, xi_1 = None, wavelet=Cauchy()):
        self.data = np.asarray(data, dtype=float)

        self.fs = fs             # Sampling frequency
        self.d = d               # Decimation factor (Uniform Decimation)
        self.b = b               # largest scale factor
        self.q = q               # frequency step factor
        self.wavelet = wavelet   # Cauchy wavelet
        self.M = M               # Number of wavelet scales
        self.Mc = Mc             # Number of compensation scales for lower frequencies
        self.xi_1 = xi_1            # Quasi-random phase factor

        # Calculate time vector, translations, and scale channels
        self.time = np.arange(0, data.shape[-1], 1 / self.fs)
        self.l_channels = np.arange(self.data.shape[-1] // self.d)
        self.j_channels = np.arange(self.Mc + self.M)

        # Calculate quasi-random scale-channel delays
        self.delta_js = self.delay(np.arange(-self.Mc, self.M))

        # TODO: calculate optimal parameters if None
        if self.q is None:
            self.q = self.wavelet.cutoff_time()


    def forward(self):  
        coeffs = np.zeros((self.l_channels.shape[0], self.j_channels.shape[0]), dtype=complex)

        for l in self.l_channels:
            for idx, j in enumerate(self.j_channels):
            
                # Note: Higher j corresponds to higher frequency
                # b is the largest scale of interest
                # q is the frequency step factor
                delta_j = self.delta_js[idx]

                if(j < -self.q/self.b):
                    # Eq 3
                    filterbank = self.wavelet.eq3()
                    coeffs[l, j] = np.sum(self.data * filterbank)
                    pass
                else:
                    # Eq 4
                    filterbank = self.wavelet.eq4()
                    coeffs[l, j] = np.sum(self.data * filterbank)
                    pass

        return(coeffs)




    def inverse(self):
        pass

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



'''