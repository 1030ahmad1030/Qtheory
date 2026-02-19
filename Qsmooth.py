"""
QSmooth: Response smoothing for persistence timescale analysis.
Pure smoothing functions only - no extra functionality.
"""

import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import gaussian_filter1d

class QSmooth:
    
    def savgol(self, t: np.ndarray, R: np.ndarray, 
               window_frac: float = 0.05,
               polyorder: int = 3) -> np.ndarray:
        """
        Savitzky-Golay smoothing.
        
        BEST FOR: τ⁽ˢ⁾, τ⁽ᵘ⁾ (derivative-critical estimators)
        PRESERVES: Local polynomial structure, peak positions, analytical derivatives
        
        Parameters:
        -----------
        t : array, time values (unused but kept for API consistency)
        R : array, response values
        window_frac : float, fraction of points in smoothing window (0.01-0.2)
        polyorder : int, polynomial order (2-4, 3=cubic recommended)
        
        Returns:
        --------
        R_smooth : smoothed response
        """
        n = len(R)
        window_length = max(5, int(window_frac * n))
        window_length = window_length + 1 if window_length % 2 == 0 else window_length
        
        if window_length > n:
            window_length = n - 1 if n % 2 == 0 else n
        
        return savgol_filter(R, window_length, polyorder)
    
    def butterworth(self, t: np.ndarray, R: np.ndarray,
                    cutoff_frac: float = 0.9,
                    order: int = 4) -> np.ndarray:
        """
        Zero-phase Butterworth filter (filtfilt).
        
        BEST FOR: τ_env², τ_E³ (frequency/phase-critical estimators)
        PRESERVES: Phase, frequency content, zero-phase distortion
        
        Parameters:
        -----------
        t : array, time values (used for dt)
        R : array, response values
        cutoff_frac : float, cutoff frequency as fraction of Nyquist (0.1-0.95)
        order : int, filter order (1-8, 4=balanced)
        
        Returns:
        --------
        R_smooth : smoothed response
        """
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt  # Sampling frequency
        nyquist = fs / 2.0
        
        # Normalize cutoff frequency (0 to 1, where 1 = Nyquist)
        cutoff_normalized = min(0.99, max(0.01, cutoff_frac))
        
        # Butterworth filter coefficients
        b, a = butter(order, cutoff_normalized, btype='low', analog=False)
        
        # Zero-phase forward-backward filtering
        return filtfilt(b, a, R)
    
    def gaussian(self, t: np.ndarray, R: np.ndarray,
                 sigma_frac: float = 0.02) -> np.ndarray:
        """
        Gaussian smoothing.
        
        BEST FOR: τ⁽²⁾, τ⁽³⁾ (shape-preserving estimators)
        PRESERVES: Global shape, monotonicity, smoothness
        
        Parameters:
        -----------
        t : array, time values (used for scaling)
        R : array, response values
        sigma_frac : float, sigma as fraction of total time span (0.005-0.1)
        
        Returns:
        --------
        R_smooth : smoothed response
        """
        t_span = t[-1] - t[0]
        sigma = sigma_frac * t_span
        
        # Convert sigma to points
        dt = np.mean(np.diff(t))
        sigma_points = sigma / dt
        
        return gaussian_filter1d(R, sigma=sigma_points)