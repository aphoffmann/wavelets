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

# Global parameters can be defined here or passed as function arguments.
# Example default wavelet parameters:
DEFAULT_W0 = 6.0
DEFAULT_XI_1 = 0.25


class MorletWavelet:
    """
    Class representing a Morlet mother wavelet and methods for generating
    channel-specific wavelets based on Equations 4 and 5 from the paper.
    """

    def __init__(self, w0=DEFAULT_W0, xi_1=DEFAULT_XI_1):
        """
        Initialize the Morlet wavelet with given parameters.
        """
        self.w0 = w0
        self.xi_1 = xi_1

    def time(self, t, s=1.0, complete=False):
        """
        Compute the Morlet wavelet in the time domain.
        
        Parameters:
        - t: Time array.
        - s: Scale.
        - complete: Include DC correction if True.
        
        Returns:
        - ψ(t) evaluated at times t.
        """
        w = self.w0
        x = t / s
        # Standard complex Morlet wavelet expression
        out = np.exp(1j * w * x)
        if complete:
            # Subtract DC component for complete Morlet wavelet formulation
            out -= np.exp(-0.5 * (w**2))
        out *= np.exp(-0.5 * (x**2)) * np.pi**(-0.25)
        return out

    def eq4(self, t, l, j, d, b, q, delta_j):
        """
        Compute Equation 4: ψ_{l,j}(t) for α_j > 0.
        
        Equation 4:
        ψ_{l,j}(t) = sqrt(1/b + j/q) * ψ((1/b + j/q) * (t - d(l + δ_j))).
        """
        alpha_j = (1.0 / b) + (j / q)
        shift = d * (l + delta_j)
        arg = alpha_j * (t - shift)
        return np.sqrt(alpha_j) * self.time(arg, s=1.0, complete=True)

    def eq5(self, t, l, j, d, b, q, delta_j):
        """
        Compute Equation 5: ψ_{l,j}^{comp}(t) for α_j ≤ 0.
        
        Equation 5:
        ψ_{l,j}^{comp}(t) = (1/√b) ψ((t - d(l+δ_j)) / b)
                          * exp(2πi ξ₁ * j (t - d(l+δ_j)) / q).
        """
        shift = d * (l + delta_j)
        tau = (t - shift) / b
        base = self.time(tau, s=1.0, complete=False)
        phase = np.exp(2j * np.pi * self.xi_1 * j * (t - shift) / q)
        return (1.0 / np.sqrt(b)) * base * phase


def generate_low_discrepancy_sequence(num_values, sequence_type='kronecker', **kwargs):
    """
    Generate a low-discrepancy sequence (e.g., Kronecker sequence) for delays.

    Parameters:
    - num_values: Number of values to generate.
    - sequence_type: Type of sequence ('kronecker', 'sobol', etc.). Currently unused.
    - kwargs: Additional parameters for sequence generation (unused).

    Returns:
    - Array of low-discrepancy values in [0, 1).
    """
    # Use the golden ratio conjugate as the irrational number
    golden_conjugate = (np.sqrt(5) - 1) / 2  
    # Generate sequence: fractional part of n*golden_conjugate for n = 1,...,num_values
    sequence = np.mod(np.arange(1, num_values + 1) * golden_conjugate, 1.0)
    return sequence



def build_analysis_filter_bank(wavelet, channels, d, b, q, delta_js, sample_rate, kernel_size):
    """
    Build analysis filters for each channel using Equations 4 and 5.

    Parameters:
    - wavelet: Instance of a wavelet class (e.g., MorletWavelet).
    - channels: List or array of channel indices j.
    - d, b, q: Grid parameters.
    - delta_js: Array of channel-specific delays.
    - sample_rate: Sampling rate.
    - kernel_size: Length of each filter kernel.

    Returns:
    - Dictionary of analysis filters keyed by channel.
    """
    filters = {}
    dt = 1.0 / sample_rate
    half = kernel_size // 2
    
    # Create a time axis centered around zero
    t = (np.arange(kernel_size) - half) * dt

    # Loop over each channel to build its filter
    for idx, j in enumerate(channels):
        # Retrieve corresponding delta_j for channel j
        delta_j = delta_js[idx] if len(delta_js) == len(channels) else delta_js[j]
        
        # Compute alpha_j to decide which equation to use
        alpha_j = (1.0 / b) + (j / q)
        
        # For simplicity, assume l = 0 for filter construction; adjust as needed
        l = 0
        
        # Use Equation 4 or Equation 5 based on the sign of alpha_j
        if alpha_j > 0:
            filter_response = wavelet.eq4(t, l, j, d, b, q, delta_j)
        else:
            filter_response = wavelet.eq5(t, l, j, d, b, q, delta_j)
        
        # Store the filter response for the current channel
        filters[j] = filter_response

    return filters



def build_synthesis_filter_bank(analysis_filters, use_multi_channel=True, dt=1.0):
    """
    Construct synthesis (dual) filters from analysis filters.

    Parameters:
    - analysis_filters: Dictionary of analysis filters keyed by channel.
    - use_multi_channel: Whether to use multi-channel dual computation.
    - dt: Time step based on sampling rate.

    Returns:
    - Dictionary of synthesis filters keyed by channel.
    """
    scale_factor = 1/np.sqrt(dt)
    
    if use_multi_channel:
        # Multi-channel dual computation using Gram matrix inversion.
        
        # Extract channel indices and prepare Gram matrix dimensions.
        channels = list(analysis_filters.keys())
        num_channels = len(channels)
        G = np.zeros((num_channels, num_channels), dtype=complex)
        
        # Build the Gram matrix G[i,j] = ⟨h_i, h_j⟩.
        for i, j1 in enumerate(channels):
            h1 = analysis_filters[j1]
            for k, j2 in enumerate(channels):
                h2 = analysis_filters[j2]
                G[i, k] = np.vdot(h1, h2)  # inner product
        
        # Invert Gram matrix.
        epsilon = 1e-8
        G += epsilon * np.eye(num_channels)
        G_inv = np.linalg.inv(G)
        
        dual_filters = {}
        # Compute dual filters using inverted Gram matrix.
        for i, j1 in enumerate(channels):
            # Initialize the dual filter for channel j1 as a zeros array.
            dual = np.zeros_like(analysis_filters[j1], dtype=complex)
            for k, j2 in enumerate(channels):
                # Use conjugate of the analysis filter for channel j2.
                dual += G_inv[i, k] * np.conjugate(analysis_filters[j2])
            # Apply scaling factor.
            dual_filters[j1] = dual * scale_factor
        
        return dual_filters

    else:
        # Naive dual filter computation: g_j = conj(h_j) / ||h_j||^2
        dual_filters = {}
        for j, h_j in analysis_filters.items():
            norm_sq = np.vdot(h_j, h_j)
            if abs(norm_sq) < 1e-14:
                # Avoid division by near-zero norm.
                dual_filters[j] = np.zeros_like(h_j)
            else:
                dual_filters[j] = (np.conjugate(h_j) / norm_sq) * scale_factor
        return dual_filters


class GridWaveletTransform:
    """
    Class to perform forward and inverse grid-based decimated wavelet transforms.
    """

    def __init__(self, data, sample_rate, b, q, d, M, M_C, xi_1, kernel_size, 
                 wavelet=None, sequence_type='kronecker'):
        """
        Initialize the transform with signal data, parameters, and wavelet.

        Parameters:
        - data: Input time-domain signal array.
        - sample_rate: Sampling rate (ξ_samp).
        - b, q, d: Grid parameters.
        - M: Desired number of frequency channels.
        - M_C: Number of compensation filters.
        - kernel_size: Length of filter kernels.
        - wavelet: Instance of a wavelet (default is MorletWavelet).
        - sequence_type: Type of low-discrepancy sequence for delays.
        """
        # Store basic parameters
        self.data = np.asarray(data, dtype=float)
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.b = b
        self.q = q
        self.d = d
        self.M = M
        self.M_C = M_C
        self.xi_1 = xi_1
        self.kernel_size = kernel_size
        
        # Initialize wavelet if not provided
        if wavelet is None:
            self.wavelet = MorletWavelet(xi_1 = self.xi_1)
        else:
            self.wavelet = wavelet
        
        # Define channel indices
        # Assuming channels from 0 to M-1 for simplicity
        self.channels = np.arange(M)
        
        # Generate low-discrepancy delays for each channel
        self.delta_js = generate_low_discrepancy_sequence(len(self.channels), sequence_type)

        # Build analysis filter bank
        self.analysis_filters = build_analysis_filter_bank(
            wavelet=self.wavelet,
            channels=self.channels,
            d=self.d, b=self.b, q=self.q,
            delta_js=self.delta_js,
            sample_rate=self.sample_rate,
            kernel_size=self.kernel_size
        )
        
        # Build synthesis filter bank
        self.synthesis_filters = build_synthesis_filter_bank(
            self.analysis_filters,
            use_multi_channel=True,
            dt=self.dt
        )

    def forward_transform(self):
        """
        Perform the forward wavelet transform on the input signal.
        
        Returns:
        - 2D NumPy array of shape (num_channels, num_samples) representing coefficients.
        """
        num_channels = len(self.channels)
        num_samples = len(self.data)
        # Initialize a 2D array to hold coefficients for each channel
        coefficients = np.zeros((num_channels, num_samples), dtype=complex)
        scale_forward = np.sqrt(self.dt)
        # Convolve input signal with each analysis filter and store in the array
        for idx, j in enumerate(self.channels):
            h_j = self.analysis_filters[j]
            coefficients[idx, :] = scale_forward * scipy.signal.fftconvolve(self.data, h_j, mode='same')
        
        return coefficients

    def inverse_transform(self, coefficients):
        """
        Perform the inverse wavelet transform given coefficients.
        
        Parameters:
        - coefficients: 2D NumPy array of shape (num_channels, num_samples)
                        representing wavelet coefficients from forward transform.
        
        Returns:
        - Reconstructed signal as a 1D NumPy array.
        """
        reconstruction = np.zeros(len(self.data), dtype=complex)
        # Iterate over each channel to convolve and sum contributions
        for idx, j in enumerate(self.channels):
            # Extract the coefficients for channel j
            coeff = coefficients[idx, :]
            g_j = self.synthesis_filters[j]
            conv_result = scipy.signal.fftconvolve(coeff, g_j, mode='same')
            reconstruction += conv_result
        return reconstruction.real

    def compute_frame_bounds(self):
        """
        (Optional) Compute frame bounds using the Gram matrix of analysis filters.
        
        Returns:
        - Lower and upper frame bounds (A, B).
        """
        channels = list(self.analysis_filters.keys())
        num_channels = len(channels)
        G = np.zeros((num_channels, num_channels), dtype=complex)
        for i, j1 in enumerate(channels):
            h1 = self.analysis_filters[j1]
            for k, j2 in enumerate(channels):
                h2 = self.analysis_filters[j2]
                G[i, k] = np.vdot(h1, h2)
        eigenvalues = np.linalg.eigvals(G)
        A = np.min(eigenvalues.real)
        B = np.max(eigenvalues.real)
        return A, B

    def plot_grid_decimation(self):
        """
        Plot the grid decimation pattern for visualization.
        The plot shows time on the x-axis and frequency channel index on the y-axis.
        Dots correspond to the center of each wavelet placement.
        """
        plt.figure(figsize=(10, 6))
        
        # For each channel, we plot points along time corresponding to integer multiples of d.
        # Frequency axis: channel indices.
        for j in self.channels:
            # Generate multiple time shifts along the length of the signal.
            # We'll assume l spans a range such that the entire signal is covered.
            num_shifts = len(self.data) // int(self.d * self.sample_rate)
            times = np.array([self.d * (l + self.delta_js[j]) for l in range(num_shifts)])  
            frequencies = np.full_like(times, j)
            plt.plot(times, frequencies, 'o', markersize=2, label=f'Channel {j}' if j==self.channels[0] else "")
        
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency Channel (j)")
        plt.title("Grid Decimation Pattern")
        plt.legend(loc='upper right', markerscale=2, fontsize='small')
        plt.grid(True)
        plt.show()

    def plot_scalogram(self, coefficients):
        """
        Plot the scalogram of the wavelet coefficients with frequency on the y-axis.

        Parameters:
        - coefficients: 2D NumPy array of shape (num_channels, num_samples)
                        containing the wavelet coefficients.
        """
        
        # Define time axis extent
        time_duration = len(self.data) / self.sample_rate
        time_extent = (0, time_duration)
        
        # Define frequency axis extent assuming linear mapping from channels to frequency
        # Map lowest channel to 0 Hz and highest channel to Nyquist frequency (sample_rate/2)
        freq_min = 0
        freq_max = self.sample_rate / 2
        freq_extent = (freq_min, freq_max)

        plt.figure(figsize=(12, 6))
        # Display the magnitude of coefficients using imshow with frequency on y-axis
        plt.imshow(np.abs(coefficients), 
                   aspect='auto', 
                   origin='lower', 
                   extent=(time_extent[0], time_extent[1], freq_extent[0], freq_extent[1]),
                   cmap='viridis')
        plt.colorbar(label='Magnitude')
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Scalogram")
        plt.show()