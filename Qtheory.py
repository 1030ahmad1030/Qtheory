"""
Qtheory: A Python library for causal environmental memory analysis across domains.
Core modules:
- QTMemoryAnalyzer: Time triangulation method
- QSMemoryAnalyzer: Space triangulation  
- QSTMemoryAnalyzer: Spacetime triangulation
- QUMemoryAnalyzer: Universal cross-domain moment analysis

Authors: Ahmad Muhammad & Dana Ali Al-Abdulmalik
Affiliation: Qatar University
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import correlate, savgol_filter, find_peaks, hilbert
from typing import Dict, List, Union, Optional, Tuple
import warnings

__version__ = "2.0.0"
__author__ = "Ahmad Muhammad, Dana Ali Al-Abdulmalik"



class QTMemoryAnalyzer:
    """
    Time-domain memory analyzer using triangulation protocol.
    Quantifies τ_memory via impulse, step, and autocorrelation methods.
    """
    
    def __init__(self, system_label="Physical System"):
        self.system_label = system_label
        self.analysis_results = {}
        self.diagnostics = {}
        self.method_equivalence = {} 

    def from_impulse_response(self, response, time):
        """Quantify memory from impulse (Green's function) data with higher moments analysis."""
        G = np.asarray(response, dtype=np.float64)
        t = np.asarray(time, dtype=np.float64)
        
        if len(G) != len(t):
            raise ValueError("Impulse response and time must have same length")
        
        valid_mask = np.isfinite(G) & np.isfinite(t)
        G = G[valid_mask]
        t = t[valid_mask]
        
        if len(G) < 10:
            raise ValueError("Insufficient valid data points")
        
        # Smoothing for better moment estimation
        if len(G) > 50:
            try:
                window_length = min(len(G) // 10 * 2 + 1, 21)
                G = savgol_filter(G, window_length, 3)
            except Exception:
                pass  # Skip smoothing if it fails
        
        G_abs = np.abs(G)
        integral_unsigned = trapezoid(G_abs, t)
        integral_signed = trapezoid(G, t)
        
        if integral_unsigned < 1e-12:
            raise ValueError("Impulse response has negligible area")
        
        # Memory distribution and first moment (τ_memory)
        memory_distribution = G_abs / integral_unsigned
        tau_unsigned = trapezoid(t * G_abs, t) / integral_unsigned
        
        if abs(integral_signed) > 1e-12:
            tau_signed = trapezoid(t * G, t) / integral_signed
        else:
            tau_signed = 0.0
        
        # OSCILLATION RATIO CALCULATION
        if tau_unsigned > 0:
            # Method 1: Compare signed vs unsigned τ
            ratio1 = min(1.0, abs(tau_signed) / tau_unsigned)
            
            # Method 2: Detect zero crossings for oscillatory behavior
            zero_crossings = np.sum(np.diff(np.sign(G)) != 0)
            is_oscillatory = zero_crossings > 2
            
            # Method 3: Check energy balance between positive and negative parts
            positive_energy = trapezoid(np.maximum(G, 0), t)
            negative_energy = trapezoid(np.maximum(-G, 0), t)
            total_energy = positive_energy + negative_energy
            
            if total_energy > 0:
                balance_ratio = min(positive_energy, negative_energy) / total_energy
            else:
                balance_ratio = 0
            
            # Combined oscillation ratio
            if is_oscillatory and balance_ratio > 0.1:
                oscillation_ratio = min(ratio1, 0.3 + 0.2 * balance_ratio)
            elif is_oscillatory:
                oscillation_ratio = min(ratio1, 0.6)
            else:
                oscillation_ratio = max(ratio1, 0.8)
        else:
            oscillation_ratio = 1.0
        
        # HIGHER MOMENTS ANALYSIS
        moments_analysis = self._compute_memory_distribution_moments(memory_distribution, t, tau_unsigned)
        
        decay_ratio = 1.0 - min(1.0, G_abs[-1] / (np.max(G_abs) + 1e-12))
        quality = decay_ratio
        
        results = {
            'tau_signed': float(tau_signed),
            'tau_unsigned': float(tau_unsigned),
            'memory_distribution': memory_distribution,
            'oscillation_ratio': float(oscillation_ratio),
            'quality': float(quality),
            'valid': True,
            'moments': moments_analysis
        }
        results['physical_meaning'] = self._interpret_impulse_results(results)
        self.analysis_results['impulse'] = results
        return results

    def _compute_memory_distribution_moments(self, memory_distribution, time, tau_memory):
        """Compute higher moments of memory distribution μ(t) for spectral characterization."""
        t = np.asarray(time)
        mu = np.asarray(memory_distribution)
        
        # Ensure normalization
        mu = mu / trapezoid(mu, t) if trapezoid(mu, t) > 0 else mu
        
        # Variance (2nd central moment) - spread of memory persistence
        variance = trapezoid((t - tau_memory)**2 * mu, t)
        std_dev = np.sqrt(variance) if variance > 0 else 0.0
        
        # Skewness (3rd standardized moment) - asymmetry of memory decay
        if std_dev > 1e-12:
            skewness = trapezoid(((t - tau_memory) / std_dev)**3 * mu, t)
        else:
            skewness = 0.0
        
        # Kurtosis (4th standardized moment) - tail behavior
        if std_dev > 1e-12:
            kurtosis = trapezoid(((t - tau_memory) / std_dev)**4 * mu, t) - 3
        else:
            kurtosis = -3.0
        
        # Multimodality detection
        modality_analysis = self._analyze_modality(mu, t)
        
        # Memory concentration metric
        concentration = tau_memory / (std_dev + 1e-12) if std_dev > 0 else float('inf')
        
        # Memory persistence types
        persistence_type = self._classify_persistence_type(skewness, kurtosis, modality_analysis)
        
        return {
            'variance': float(variance),
            'std_dev': float(std_dev),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'concentration': float(concentration),
            'modality': modality_analysis,
            'persistence_type': persistence_type,
            'interpretation': self._interpret_moments(skewness, kurtosis, modality_analysis)
        }

    def _analyze_modality(self, memory_distribution, time):
        """Analyze multimodality of memory distribution."""
        mu = memory_distribution
        t = time
        
        # Improved peak detection for multi-exponential systems
        peak_height_threshold = np.max(mu) * 0.05
        min_distance = max(1, len(mu) // 50)
        
        peaks, properties = find_peaks(mu, height=peak_height_threshold, distance=min_distance)
        
        n_peaks = len(peaks)
        peak_times = t[peaks] if n_peaks > 0 else np.array([])
        peak_heights = mu[peaks] if n_peaks > 0 else np.array([])
        
        if n_peaks == 0:
            modality_type = "NO_CLEAR_PEAKS"
        elif n_peaks == 1:
            modality_type = "UNIMODAL"
        elif n_peaks == 2:
            modality_type = "BIMODAL"
        else:
            modality_type = "MULTIMODAL"
        
        peak_separation = np.diff(peak_times) if len(peak_times) > 1 else np.array([])
        avg_separation = np.mean(peak_separation) if len(peak_separation) > 0 else 0.0
        
        return {
            'n_peaks': n_peaks,
            'peak_times': peak_times.tolist(),
            'peak_heights': peak_heights.tolist(),
            'modality_type': modality_type,
            'peak_separation': peak_separation.tolist(),
            'avg_separation': float(avg_separation)
        }

    def _classify_persistence_type(self, skewness, kurtosis, modality_analysis):
        """Classify memory persistence type based on moment analysis."""
        n_peaks = modality_analysis['n_peaks']
        
        if n_peaks > 1:
            return "MULTIPROCESS"
        
        if abs(skewness) < 0.5:
            if kurtosis < -1:
                return "SHARP_DECAY"
            elif kurtosis < 1:
                return "EXPONENTIAL_LIKE"
            else:
                return "LONG_TAILED"
        elif skewness > 0.5:
            return "SLOW_TAIL"
        else:
            return "FAST_INITIAL"

    def _interpret_moments(self, skewness, kurtosis, modality_analysis):
        """Generate physical interpretation of higher moments."""
        interpretations = []
        
        n_peaks = modality_analysis['n_peaks']
        modality_type = modality_analysis['modality_type']
        
        if modality_type == "BIMODAL":
            interpretations.append("Multiple memory processes with distinct timescales")
        elif modality_type == "MULTIMODAL":
            interpretations.append("Complex memory system with multiple persistence mechanisms")
        
        if skewness > 0.5:
            interpretations.append("Slow memory decay with persistent tail effects")
        elif skewness < -0.5:
            interpretations.append("Rapid initial memory loss followed by slower decay")
        else:
            interpretations.append("Symmetric memory decay profile")
        
        if kurtosis > 1:
            interpretations.append("Heavy-tailed memory distribution (outlier persistence)")
        elif kurtosis < -1:
            interpretations.append("Light-tailed distribution (sharp memory cutoff)")
        else:
            interpretations.append("Moderate tail behavior")
        
        return "; ".join(interpretations) if interpretations else "Standard memory persistence"

    def _interpret_impulse_results(self, results):
        """Enhanced interpretation with higher moments insights."""
        tau_signed = results['tau_signed']
        tau_unsigned = results['tau_unsigned']
        osc_ratio = results['oscillation_ratio']
        moments = results.get('moments', {})
        
        base_interpretation = ""
        if osc_ratio > 0.7:
            base_interpretation = f"Monotonic decay system: τ = {tau_unsigned:.3f} represents average decay time"
        elif osc_ratio > 0.4:
            base_interpretation = f"Moderately oscillatory: τ_unsigned = {tau_unsigned:.3f} (persistence), τ_signed = {tau_signed:.3f} (net effect)"
        else:
            base_interpretation = f"Strongly oscillatory: τ_unsigned = {tau_unsigned:.3f} (energy decay), τ_signed = {tau_signed:.3f} (sign cancellation)"
        
        if moments:
            persistence_type = moments.get('persistence_type', 'UNKNOWN')
            modality_type = moments.get('modality', {}).get('modality_type', 'UNIMODAL')
            
            if modality_type != "UNIMODAL":
                base_interpretation += f" | {modality_type} memory"
            if persistence_type != "EXPONENTIAL_LIKE":
                base_interpretation += f" | {persistence_type.replace('_', ' ').title()}"
        
        return base_interpretation


    def from_step_response(self, response, time):
        """Quantify memory from step input transient data - PAPER METHOD 2."""
        R = np.asarray(response, dtype=np.float64)
        t = np.asarray(time, dtype=np.float64)
        
        if len(R) != len(t):
            raise ValueError("Step response and time must have same length")
        
        valid_mask = np.isfinite(R) & np.isfinite(t)
        R = R[valid_mask]
        t = t[valid_mask]
        
        if len(R) < 10:
            raise ValueError("Insufficient valid data points")
        
        steady_start = len(R) * 9 // 10
        R_inf = np.mean(R[steady_start:])
        if abs(R_inf - R[0]) < 1e-12:
            raise ValueError("No significant step change detected")
        
        # PAPER METHOD 2: Step Response Integration
        R_norm = (R - R[0]) / (R_inf - R[0])
        
        # Detect oscillatory behavior
        derivative = np.diff(R_norm)
        sign_changes = np.sum(np.diff(np.sign(derivative)) != 0)
        is_oscillatory = sign_changes > 3
        
        # PAPER: For oscillatory systems, use absolute deviation
        if is_oscillatory:
            deviation = np.abs(1 - R_norm)
            method_used = "absolute_deviation"
        else:
            deviation = np.maximum(1 - R_norm, 0)
            method_used = "standard"
            
        tau_memory = trapezoid(deviation, t)
        
        # PAPER'S METHOD EQUIVALENCE VALIDATION
        if 'impulse' in self.analysis_results:
            tau_impulse = self.analysis_results['impulse']['tau_signed']
            equivalence_error = abs(tau_memory - tau_impulse) / max(abs(tau_memory), abs(tau_impulse))
            self.diagnostics['method_1_2_equivalence'] = {
                'error': equivalence_error,
                'lti_valid': equivalence_error < 0.05,
                'paper_theorem': "Method 1 ≡ Method 2 for LTI systems"
            }
        
        observation_time = t[-1] - t[0]
        if tau_memory > observation_time * 0.8:
            raise ValueError(f"τ too large for observation window")
        
        results = {
            'tau_memory': float(tau_memory),
            'method_used': method_used,
            'is_oscillatory': is_oscillatory,
            'steady_state': float(R_inf),
            'valid': True
        }
        results['physical_meaning'] = self._interpret_step_results(results)
        self.analysis_results['step'] = results
        
        # Store for triangulation
        self.method_equivalence['step'] = tau_memory
        
        return results

    def from_time_series(self, signal, time):
        """Quantify memory from stationary time-series via autocorrelation - PAPER METHOD 3."""
        if signal is None:
            raise ValueError("No time series data provided")
        
        X = np.asarray(signal, dtype=np.float64)
        t = np.asarray(time, dtype=np.float64)
        
        if len(X) != len(t):
            raise ValueError("Time series and time must have same length")
        
        valid_mask = np.isfinite(X) & np.isfinite(t)
        X = X[valid_mask]
        t = t[valid_mask]
        
        if len(X) < 50:
            raise ValueError("Insufficient data for autocorrelation analysis")
        
        # PAPER: Stationarity check for valid autocorrelation
        n_segments = 5
        segment_size = len(X) // n_segments
        means = [np.mean(X[i*segment_size:(i+1)*segment_size]) for i in range(n_segments)]
        mean_std = np.std(means) / (np.std(X) + 1e-12)
        is_stationary = mean_std < 0.2
        
        if not is_stationary:
            raise ValueError(f"Time series is non-stationary (mean_std={mean_std:.2f})")
        
        # PAPER METHOD 3: Autocorrelation Analysis
        X_centered = X - np.mean(X)
        autocorr = correlate(X_centered, X_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr[:min(len(autocorr), len(X_centered)//2)]
        
        if autocorr[0] <= 1e-12:
            raise ValueError("Zero autocorrelation")
        autocorr_norm = autocorr / autocorr[0]
        
        dt = t[1] - t[0] if len(t) > 1 else 1.0
        lags = np.arange(len(autocorr_norm)) * dt
        
        # PAPER: Integration cutoff
        threshold_idx = np.where(autocorr_norm < 0.01)[0]
        end_idx = threshold_idx[0] if len(threshold_idx) > 0 else len(autocorr_norm)
        end_idx = min(end_idx, len(autocorr_norm))
        
        autocorr_norm = autocorr_norm[:end_idx]
        lags = lags[:end_idx]
        tau_memory = trapezoid(autocorr_norm, lags)
        
        # PAPER'S METHOD EQUIVALENCE: τ3 ≡ τ1 only for exponential responses
        if 'impulse' in self.analysis_results:
            tau_impulse = self.analysis_results['impulse']['tau_unsigned']
            is_exponential = self._check_exponential_response()
            
            self.diagnostics['method_1_3_equivalence'] = {
                'is_exponential': is_exponential,
                'paper_theorem': "Method 3 ≡ Method 1 only for exponential responses",
                'tau_ratio': tau_memory / tau_impulse if tau_impulse > 0 else float('inf')
            }
        
        if tau_memory < dt * 2:
            raise ValueError(f"τ too small ({tau_memory:.3f})")
        
        results = {
            'tau_memory': float(tau_memory),
            'autocorr_function': autocorr_norm,
            'lags': lags,
            'is_stationary': is_stationary,
            'data_variance': float(np.var(X)),
            'valid': True
        }
        results['physical_meaning'] = self._interpret_autocorr_results(results)
        self.analysis_results['autocorr'] = results
        
        # Store for triangulation
        self.method_equivalence['autocorr'] = tau_memory
        
        return results



    def _interpret_step_results(self, results):
        """Interpret step response results - WAS MISSING"""
        tau = results['tau_memory']
        oscillatory = results['is_oscillatory']
        if oscillatory:
            return f"Oscillatory settling: τ = {tau:.3f} (total area of deviation from steady state)"
        else:
            return f"Monotonic settling: τ = {tau:.3f} (total lag area to reach steady state)"

    def _interpret_autocorr_results(self, results):
        """Interpret autocorrelation results - WAS MISSING"""
        tau = results['tau_memory']
        return f"Statistical memory: τ = {tau:.3f} (predictability horizon for stationary process)"

    def get_memory_spectrum_report(self):
        """Generate comprehensive memory spectrum report - WAS MISSING"""
        if 'impulse' not in self.analysis_results:
            raise ValueError("No impulse response analysis available")
        
        moments = self.analysis_results['impulse'].get('moments', {})
        modality = moments.get('modality', {})
        
        report = {
            'spectral_summary': {
                'persistence_type': moments.get('persistence_type', 'UNKNOWN'),
                'modality': modality.get('modality_type', 'UNKNOWN'),
                'spread_ratio': moments.get('concentration', 0.0),
                'temporal_complexity': 'HIGH' if modality.get('n_peaks', 0) > 1 else 'LOW'
            },
            'moment_breakdown': {
                'variance': f"{moments.get('variance', 0):.4f} (spread: ±{moments.get('std_dev', 0):.3f})",
                'skewness': f"{moments.get('skewness', 0):.3f}",
                'kurtosis': f"{moments.get('kurtosis', 0):.3f}",
                'interpretation': moments.get('interpretation', 'No interpretation')
            },
            'modality_analysis': {
                'peak_count': modality.get('n_peaks', 0),
                'peak_times': modality.get('peak_times', []),
                'dominant_process': f"τ = {modality.get('peak_times', [0])[0]:.3f}" if modality.get('peak_times') else "None"
            }
        }
        
        return report

    def full_analysis(self, impulse=None, step=None, signal=None, time=None):
        """
        Run all applicable methods and return complete memory characterization.
        Only requires the data modalities you have (flexible input).
        """
        if time is None:
            raise ValueError("Time vector must be provided")
        
        if impulse is not None:
            try:
                self.from_impulse_response(impulse, time)
            except Exception as e:
                print(f"Method 1 (Impulse) failed: {e}")
        
        if step is not None:
            try:
                self.from_step_response(step, time)
            except Exception as e:
                print(f"Method 2 (Step) failed: {e}")
        
        if signal is not None:
            try:
                self.from_time_series(signal, time)
            except Exception as e:
                print(f"Method 3 (Time Series) failed: {e}")
        
        self.triangulate_memory()
        return self.generate_report()

    def generate_report(self):
        """Return structured, interpretable summary of memory characteristics - WAS MISSING"""
        if 'triangulation' not in self.analysis_results:
            self.triangulate_memory()
        
        report = {
            'system_label': self.system_label,
            'tau_impulse': self.analysis_results.get('impulse', {}).get('tau_unsigned'),
            'tau_step': self.analysis_results.get('step', {}).get('tau_memory'),
            'tau_autocorr': self.analysis_results.get('autocorr', {}).get('tau_memory'),
            'tau_consensus': self.analysis_results['triangulation']['tau_consensus'],
            'memory_distribution': self.analysis_results.get('impulse', {}).get('memory_distribution'),
            'tau_signed': self.analysis_results.get('impulse', {}).get('tau_signed'),
            'tau_unsigned': self.analysis_results.get('impulse', {}).get('tau_unsigned'),
            'system_type': self.analysis_results['triangulation']['system_type'],
            'method_agreement': self.analysis_results['triangulation']['method_agreement'],
            'confidence': self.analysis_results['triangulation']['confidence'],
            'characteristics': self.analysis_results['triangulation']['characteristics'],
            'physical_interpretations': {
                'impulse': self.analysis_results.get('impulse', {}).get('physical_meaning'),
                'step': self.analysis_results.get('step', {}).get('physical_meaning'),
                'autocorr': self.analysis_results.get('autocorr', {}).get('physical_meaning')
            }
        }
        
        try:
            spectral_report = self.get_memory_spectrum_report()
            report['spectral_analysis'] = spectral_report
        except Exception:
            report['spectral_analysis'] = "Not available"
            
        return report

    # PAPER SECTION 3.1: Generalization to Unstable Systems - WAS MISSING
    def windowed_memory_analysis(self, response, time, t_max=None):
        """PAPER EQ 13: Windowed memory timescale for finite observation."""
        if t_max is None:
            t_max = time[-1]
        
        window_mask = time <= t_max
        G_window = response[window_mask]
        t_window = time[window_mask]
        
        return self.from_impulse_response(G_window, t_window)

    def envelope_effective_memory(self, response, time):
        """PAPER: Envelope-based effective memory for oscillatory systems."""
        from scipy.signal import hilbert
        
        analytic_signal = hilbert(response)
        envelope = np.abs(analytic_signal)
        
        # Fit exponential to envelope
        try:
            from scipy.optimize import curve_fit
            
            def exp_decay(t, a, b, c):
                return a * np.exp(-b * t) + c
            
            popt, pcov = curve_fit(exp_decay, time, envelope, 
                                 p0=[envelope[0], 1/time[-1], envelope[-1]])
            
            alpha = popt[1]
            if alpha > 0:
                tau_eff = 1 / alpha
            else:
                tau_eff = time[-1]  # Marginally stable
                
            return {
                'tau_effective': tau_eff,
                'envelope': envelope,
                'growth_rate': alpha,
                'stability': 'stable' if alpha > 0 else 'unstable' if alpha < 0 else 'marginal'
            }
        except:
            return {'tau_effective': time[-1], 'envelope': envelope, 'stability': 'unknown'}
    def _check_exponential_response(self):
        """Check if impulse response is exponential (for Method 3 equivalence validation)."""
        if 'impulse' not in self.analysis_results:
            return False
            
        G = self.analysis_results['impulse'].get('raw_response', None)
        if G is None:
            return False
            
        # Simple exponential fit check
        try:
            from scipy.optimize import curve_fit
            
            def exp_func(t, a, b):
                return a * np.exp(-b * t)
            
            t = np.arange(len(G))
            popt, pcov = curve_fit(exp_func, t, np.abs(G), maxfev=1000)
            
            # Goodness of fit
            fitted = exp_func(t, *popt)
            r_squared = 1 - np.sum((G - fitted)**2) / np.sum((G - np.mean(G))**2)
            
            return r_squared > 0.95
        except:
            return False

    def triangulate_memory(self):
        """Fuse results from available methods - PAPER'S TRIANGULATION PROTOCOL."""
        if not self.analysis_results:
            raise ValueError("No analysis results available for triangulation")
        
        # PAPER: Extract all available τ values
        tau_impulse_signed = self.analysis_results['impulse']['tau_signed'] \
            if 'impulse' in self.analysis_results and self.analysis_results['impulse']['valid'] else None
        tau_impulse_unsigned = self.analysis_results['impulse']['tau_unsigned'] \
            if 'impulse' in self.analysis_results and self.analysis_results['impulse']['valid'] else None
        tau_step = self.analysis_results['step']['tau_memory'] \
            if 'step' in self.analysis_results and self.analysis_results['step']['valid'] else None
        tau_autocorr = self.analysis_results['autocorr']['tau_memory'] \
            if 'autocorr' in self.analysis_results and self.analysis_results['autocorr']['valid'] else None
        
        valid_taus = [tau for tau in [tau_impulse_unsigned, tau_step, tau_autocorr] if tau is not None]
        if not valid_taus:
            raise ValueError("No valid τ values for triangulation")
        
        # PAPER: Consensus τ_memory
        tau_consensus = np.mean(valid_taus)
        
        # PAPER: Method agreement analysis
        method_agreement = self._analyze_method_agreement(tau_impulse_unsigned, tau_step, tau_autocorr)
        
        # PAPER TABLE 2: Diagnostic interpretation
        system_type, confidence, characteristics = self._characterize_system_diagnostics(
            tau_impulse_signed, tau_impulse_unsigned, tau_step, tau_autocorr, method_agreement)
        
        triangulation_results = {
            'tau_impulse_signed': tau_impulse_signed,
            'tau_impulse_unsigned': tau_impulse_unsigned,
            'tau_step': tau_step,
            'tau_autocorr': tau_autocorr,
            'tau_consensus': float(tau_consensus),
            'method_agreement': method_agreement,
            'system_type': system_type,
            'confidence': confidence,
            'characteristics': characteristics,
            'valid_methods_count': len(valid_taus),
            # PAPER: Key diagnostic patterns
            'signed_unsigned_ratio': tau_impulse_signed / tau_impulse_unsigned if tau_impulse_unsigned else None,
            'oscillatory_behavior': tau_impulse_signed and tau_impulse_unsigned and abs(tau_impulse_signed) < 0.5 * tau_impulse_unsigned
        }
        
        self.analysis_results['triangulation'] = triangulation_results
        return triangulation_results

    def _analyze_method_agreement(self, tau1, tau2, tau3):
        """PAPER: Method agreement quantification."""
        valid_taus = [tau for tau in [tau1, tau2, tau3] if tau is not None]
        if len(valid_taus) < 2:
            return "INSUFFICIENT_METHODS"
        
        mean_tau = np.mean(valid_taus)
        std_tau = np.std(valid_taus)
        consistency = 1 - (std_tau / mean_tau) if mean_tau > 0 else 0
        
        # PAPER: Agreement thresholds
        if consistency > 0.9:
            return "EXCELLENT"
        elif consistency > 0.7:
            return "GOOD" 
        elif consistency > 0.5:
            return "MODERATE"
        else:
            return "POOR"

    def _characterize_system_diagnostics(self, tau_signed, tau_unsigned, tau_step, tau_autocorr, method_agreement):
        """PAPER TABLE 2: System characterization from method patterns."""
        available_methods = sum(1 for tau in [tau_unsigned, tau_step, tau_autocorr] if tau is not None)
        
        if available_methods < 2:
            return "INSUFFICIENT_DATA", "LOW", ["Need at least 2 methods for reliable classification"]
        
        oscillation_ratio = self.analysis_results.get('impulse', {}).get('oscillation_ratio', 1.0)
        modality_info = self.analysis_results.get('impulse', {}).get('moments', {}).get('modality', {})
        n_peaks = modality_info.get('n_peaks', 0)
        
        # PAPER TABLE 2: Diagnostic patterns
        if tau_signed and tau_unsigned and abs(tau_signed) < 0.3 * tau_unsigned:
            # Strong oscillatory behavior
            return "STRONGLY_OSCILLATORY", "HIGH", [
                "Strong wave interference effects",
                "Sign cancellation in causal influence",
                "Wave-like memory persistence"
            ]
        
        elif method_agreement == "EXCELLENT" and oscillation_ratio > 0.7:
            # PAPER: Well-behaved LTI system
            return "WELL_BEHAVED_LTI", "HIGH", [
                "Linear time-invariant system confirmed",
                "Method equivalence theorems hold",
                "Exponential-like memory decay"
            ]
        
        elif tau_autocorr and tau_unsigned and tau_autocorr < 0.5 * tau_unsigned:
            # PAPER: Oscillatory memory pattern
            return "OSCILLATORY_MEMORY", "HIGH", [
                "τ₁ ≈ τ₂ ≫ τ₃ pattern detected",
                "Sign cancellation in autocorrelation",
                "Oscillatory environmental memory"
            ]
        
        elif method_agreement == "POOR" and 'autocorr' not in self.analysis_results:
            # PAPER: Non-stationary process
            return "NON_STATIONARY", "HIGH", [
                "Invalid autocorrelation method",
                "Time-varying statistics",
                "Non-stationary environmental process"
            ]
        
        elif n_peaks > 1:
            # PAPER: Multiple memory processes
            return "MULTIPROCESS_MEMORY", "MEDIUM_HIGH", [
                f"Multiple memory timescales ({n_peaks} processes)",
                "Complex memory distribution",
                "Overlapping persistence mechanisms"
            ]
        
        else:
            return "COMPLEX_SYSTEM", "MEDIUM", [
                "Non-standard memory characteristics",
                "Further investigation recommended",
                "Check measurement conditions"
            ]

    def validate_paper_theorems(self):
        """PAPER SECTION 2.5: Validate mathematical equivalence theorems."""
        theorems = {}
        
        # Theorem 1: Method 1 ≡ Method 2 for LTI systems
        if 'impulse' in self.analysis_results and 'step' in self.analysis_results:
            tau1 = self.analysis_results['impulse']['tau_signed']
            tau2 = self.analysis_results['step']['tau_memory']
            error_1_2 = abs(tau1 - tau2) / max(abs(tau1), abs(tau2))
            theorems['method_1_2_equivalence'] = {
                'theorem': "τ₁ ≡ τ₂ for stable LTI systems",
                'holds': error_1_2 < 0.05,
                'error': error_1_2,
                'interpretation': "LTI assumption valid" if error_1_2 < 0.05 else "Nonlinearity/time-variance detected"
            }
        
        # Theorem 2: Method 3 ≡ Method 1 only for exponential responses
        if 'impulse' in self.analysis_results and 'autocorr' in self.analysis_results:
            is_exponential = self._check_exponential_response()
            tau1 = self.analysis_results['impulse']['tau_unsigned']
            tau3 = self.analysis_results['autocorr']['tau_memory']
            error_1_3 = abs(tau1 - tau3) / max(abs(tau1), abs(tau3)) if tau1 > 0 else float('inf')
            
            theorems['method_1_3_equivalence'] = {
                'theorem': "τ₃ ≡ τ₁ only for exponential impulse responses",
                'holds': is_exponential and error_1_3 < 0.05,
                'is_exponential': is_exponential,
                'error': error_1_3,
                'interpretation': "Exponential system" if is_exponential else "Non-exponential response"
            }
        
        return theorems

    def generate_paper_report(self):
        """Generate report following paper's structure and terminology."""
        if 'triangulation' not in self.analysis_results:
            self.triangulate_memory()
        
        theorems = self.validate_paper_theorems()
        
        report = {
            'paper_reference': "A Unified Framework for Quantifying Environmental Memory in Physical Systems",
            'system_label': self.system_label,
            
            # PAPER SECTION 2.2: Memory Timescales
            'memory_timescales': {
                'tau_signed': self.analysis_results.get('impulse', {}).get('tau_signed'),
                'tau_unsigned': self.analysis_results.get('impulse', {}).get('tau_unsigned'),
                'signed_unsigned_ratio': self.analysis_results['triangulation'].get('signed_unsigned_ratio'),
                'oscillation_detected': self.analysis_results['triangulation'].get('oscillatory_behavior')
            },
            
            # PAPER SECTION 2.3: Memory Distribution
            'memory_distribution': {
                'available': 'impulse' in self.analysis_results,
                'higher_moments': self.analysis_results.get('impulse', {}).get('moments', {}),
                'spectral_characterization': self.get_memory_spectrum_report()
            },
            
            # PAPER SECTION 2.4: Triangulation Results
            'triangulation_results': self.analysis_results['triangulation'],
            
            # PAPER SECTION 2.5: Method Equivalence Theorems
            'mathematical_theorems': theorems,
            
            # PAPER TABLE 2: Diagnostic Interpretation
            'diagnostic_interpretation': {
                'system_type': self.analysis_results['triangulation']['system_type'],
                'confidence': self.analysis_results['triangulation']['confidence'],
                'characteristics': self.analysis_results['triangulation']['characteristics'],
                'method_agreement': self.analysis_results['triangulation']['method_agreement']
            }
        }
        
        return report




        
#............................................................................................................................................................

class QSMemoryAnalyzer:
    """
    Spatial memory analyzer for spatial triangulation and moment analysis.
    Extends time-domain concepts to spatial domains.
    """
    
    def __init__(self, system_label="Spatial System"):
        self.system_label = system_label
        self.analysis_results = {}

    def from_spatial_response(self, response, coordinates):
        """
        Analyze spatial memory from spatial response function.
        
        Parameters:
        -----------
        response : array-like
            Spatial response field (e.g., concentration, intensity, potential)
        coordinates : array-like or tuple of arrays
            Spatial coordinates (1D, 2D, or 3D)
        """
        field = np.asarray(response, dtype=np.float64)
        
        if isinstance(coordinates, (list, tuple)) and len(coordinates) > 1:
            # Multi-dimensional case
            return self._analyze_spatial_field_nd(field, coordinates)
        else:
            # 1D case
            coords = np.asarray(coordinates, dtype=np.float64)
            return self._analyze_spatial_1d(field, coords)

    def _analyze_spatial_1d(self, field, coordinates):
        """Analyze 1D spatial memory distribution."""
        f_abs = np.abs(field)
        total_influence = trapezoid(f_abs, coordinates)
        
        if total_influence < 1e-12:
            raise ValueError("Spatial response has negligible integrated influence")
        
        # Spatial memory distribution
        spatial_distribution = f_abs / total_influence
        spatial_centroid = trapezoid(coordinates * spatial_distribution, coordinates)
        
        # Spatial variance and extent
        spatial_variance = trapezoid((coordinates - spatial_centroid)**2 * spatial_distribution, coordinates)
        spatial_std = np.sqrt(spatial_variance) if spatial_variance > 0 else 0.0
        
        # Higher moments
        if spatial_std > 1e-12:
            spatial_skewness = trapezoid(((coordinates - spatial_centroid) / spatial_std)**3 * spatial_distribution, coordinates)
            spatial_kurtosis = trapezoid(((coordinates - spatial_centroid) / spatial_std)**4 * spatial_distribution, coordinates) - 3
        else:
            spatial_skewness = 0.0
            spatial_kurtosis = -3.0
        
        # Modality analysis
        modality = self._analyze_spatial_modality(spatial_distribution, coordinates)
        
        results = {
            'spatial_centroid': float(spatial_centroid),
            'spatial_variance': float(spatial_variance),
            'spatial_std': float(spatial_std),
            'spatial_skewness': float(spatial_skewness),
            'spatial_kurtosis': float(spatial_kurtosis),
            'total_influence': float(total_influence),
            'spatial_distribution': spatial_distribution,
            'modality': modality,
            'spatial_scale': float(spatial_std),
            'interpretation': self._interpret_spatial_results(spatial_centroid, spatial_std, modality)
        }
        
        self.analysis_results['spatial'] = results
        return results

    def _analyze_spatial_field_nd(self, field, coordinates):
        """Analyze multi-dimensional spatial fields."""
        results = {}
        dim = len(coordinates)
        
        for i in range(dim):
            # Create marginal distribution along axis i
            sum_axes = tuple(j for j in range(dim) if j != i)
            marginal = np.sum(np.abs(field), axis=sum_axes)
            coord = coordinates[i]
            
            results[f'axis_{i}'] = self._analyze_spatial_1d(marginal, coord)
        
        # Combined spatial characteristics
        centroids = [r['spatial_centroid'] for r in results.values()]
        spreads = [r['spatial_std'] for r in results.values()]
        
        results['combined'] = {
            'centroid_vector': centroids,
            'spread_vector': spreads,
            'effective_radius': np.sqrt(sum(s**2 for s in spreads)),
            'aspect_ratio': max(spreads) / min(spreads) if min(spreads) > 0 else float('inf')
        }
        
        self.analysis_results['spatial_nd'] = results
        return results

    def _analyze_spatial_modality(self, distribution, coordinates):
        """Analyze spatial modality (peaks in spatial distribution)."""
        peaks, properties = find_peaks(distribution, height=np.max(distribution)*0.1, 
                                     distance=len(distribution)//10)
        
        n_peaks = len(peaks)
        peak_locations = coordinates[peaks] if n_peaks > 0 else []
        peak_heights = distribution[peaks] if n_peaks > 0 else []
        
        if n_peaks == 0:
            modality_type = "UNIFORM"
        elif n_peaks == 1:
            modality_type = "CENTRALIZED"
        elif n_peaks == 2:
            modality_type = "BIPOLAR"
        else:
            modality_type = "MULTIPLE_HOTSPOTS"
        
        return {
            'n_peaks': n_peaks,
            'peak_locations': peak_locations.tolist(),
            'peak_heights': peak_heights.tolist(),
            'modality_type': modality_type
        }

    def _interpret_spatial_results(self, centroid, std, modality):
        """Interpret spatial memory characteristics."""
        modality_type = modality['modality_type']
        n_peaks = modality['n_peaks']
        
        base = f"Spatial center: {centroid:.3f}, Spread: {std:.3f}"
        
        if modality_type == "UNIFORM":
            return base + " | Uniform spatial influence"
        elif modality_type == "CENTRALIZED":
            return base + " | Centralized spatial memory"
        elif modality_type == "BIPOLAR":
            return base + " | Bipolar spatial distribution"
        elif modality_type == "MULTIPLE_HOTSPOTS":
            return base + f" | Multiple hotspots ({n_peaks} centers)"
        
        return base

    def spatial_triangulation(self, responses, coordinates_list):
        """
        Perform spatial triangulation using multiple response measurements.
        
        Parameters:
        -----------
        responses : list of arrays
            Multiple spatial response measurements
        coordinates_list : list of coordinate arrays
            Corresponding coordinates for each response
        """
        if len(responses) != len(coordinates_list):
            raise ValueError("Number of responses must match number of coordinate sets")
        
        triangulation_results = {}
        
        for i, (response, coords) in enumerate(zip(responses, coordinates_list)):
            triangulation_results[f'measurement_{i}'] = self.from_spatial_response(response, coords)
        
        # Compute spatial agreement
        if len(triangulation_results) > 1:
            centroids = [r['spatial_centroid'] for r in triangulation_results.values()]
            spreads = [r['spatial_std'] for r in triangulation_results.values()]
            
            triangulation_results['agreement'] = {
                'mean_centroid': np.mean(centroids),
                'std_centroid': np.std(centroids),
                'mean_spread': np.mean(spreads),
                'std_spread': np.std(spreads),
                'spatial_consistency': 'high' if np.std(centroids) < 0.1 * np.mean(spreads) else 'low'
            }
        
        self.analysis_results['spatial_triangulation'] = triangulation_results
        return triangulation_results

#..........................................................................................................................................................
class QSTMemoryAnalyzer:
    """
    Spacetime memory analyzer combining temporal and spatial analysis.
    Unified framework for spatio-temporal causal memory quantification.
    """
    
    def __init__(self, system_label="SpatioTemporal System"):
        self.system_label = system_label
        self.temporal_analyzer = QTMemoryAnalyzer(system_label + " (Temporal)")
        self.spatial_analyzer = QSMemoryAnalyzer(system_label + " (Spatial)")
        self.analysis_results = {}

    def from_spatiotemporal_field(self, field, time, spatial_coords):
        """
        Analyze spatio-temporal memory from field data.
        
        Parameters:
        -----------
        field : 2D+ array
            Spatio-temporal field [time, spatial_dims...]
        time : array-like
            Temporal coordinates
        spatial_coords : tuple of arrays
            Spatial coordinates for each dimension
        """
        field = np.asarray(field)
        time = np.asarray(time)
        
        # Temporal analysis at each spatial point
        temporal_results = {}
        spatial_shape = field.shape[1:]
        
        # For simplicity, analyze spatial average first
        if len(spatial_shape) == 1:
            # 1D space
            spatial_avg = np.mean(field, axis=1)
        else:
            # Higher dimensional space - flatten spatial dimensions
            spatial_avg = np.mean(field.reshape(field.shape[0], -1), axis=1)
        
        temporal_analysis = self.temporal_analyzer.full_analysis(signal=spatial_avg, time=time)
        
        # Spatial analysis at characteristic times
        if 'impulse' in self.temporal_analyzer.analysis_results:
            tau_characteristic = self.temporal_analyzer.analysis_results['impulse']['tau_unsigned']
            # Find time index closest to characteristic time
            time_idx = np.argmin(np.abs(time - tau_characteristic))
            spatial_snapshot = field[time_idx]
            
            spatial_analysis = self.spatial_analyzer.from_spatial_response(
                spatial_snapshot, spatial_coords)
        else:
            spatial_analysis = None
        
        # Spatio-temporal coupling analysis
        coupling_analysis = self._analyze_spatiotemporal_coupling(field, time, spatial_coords)
        
        results = {
            'temporal': temporal_analysis,
            'spatial': spatial_analysis,
            'coupling': coupling_analysis,
            'characteristic_time': float(tau_characteristic) if 'tau_characteristic' in locals() else None,
            'spatio_temporal_scale': self._compute_spatiotemporal_scale(temporal_analysis, spatial_analysis)
        }
        
        self.analysis_results['spatiotemporal'] = results
        return results

    def _analyze_spatiotemporal_coupling(self, field, time, spatial_coords):
        """Analyze coupling between spatial and temporal memory."""
        # Compute spatial variance over time
        spatial_variance_time = []
        for t_idx in range(len(time)):
            snapshot = field[t_idx]
            if len(snapshot.shape) == 1:
                # 1D spatial field
                spatial_variance = np.var(snapshot)
            else:
                # Multi-dimensional - use flattened version
                spatial_variance = np.var(snapshot.flatten())
            spatial_variance_time.append(spatial_variance)
        
        spatial_variance_time = np.array(spatial_variance_time)
        
        # Analyze how spatial structure evolves over time
        if len(spatial_variance_time) > 10:
            # Fit exponential decay to spatial variance
            try:
                from scipy.optimize import curve_fit
                
                def exp_decay(t, a, b, c):
                    return a * np.exp(-b * t) + c
                
                popt, pcov = curve_fit(exp_decay, time, spatial_variance_time, 
                                     p0=[spatial_variance_time[0], 1/time[-1], spatial_variance_time[-1]])
                
                coupling_time = 1 / popt[1]  # Characteristic coupling time
                coupling_strength = (popt[0]) / (popt[0] + popt[2])  # Relative strength
                
            except Exception:
                coupling_time = time[-1] / 2
                coupling_strength = 0.5
        else:
            coupling_time = time[-1] / 2
            coupling_strength = 0.5
        
        return {
            'coupling_time': float(coupling_time),
            'coupling_strength': float(coupling_strength),
            'spatial_variance_evolution': spatial_variance_time.tolist(),
            'coupling_type': 'strong' if coupling_strength > 0.7 else 'moderate' if coupling_strength > 0.3 else 'weak'
        }

    def _compute_spatiotemporal_scale(self, temporal_analysis, spatial_analysis):
        """Compute combined spatio-temporal scale metric."""
        if temporal_analysis and spatial_analysis:
            tau = temporal_analysis.get('tau_consensus', 1.0)
            spatial_scale = spatial_analysis.get('spatial_std', 1.0)
            
            # Combined scale (could be interpreted as propagation speed)
            if tau > 0 and spatial_scale > 0:
                characteristic_speed = spatial_scale / tau
            else:
                characteristic_speed = 0.0
            
            return {
                'temporal_scale': tau,
                'spatial_scale': spatial_scale,
                'characteristic_speed': characteristic_speed,
                'spatio_temporal_ratio': spatial_scale / tau if tau > 0 else float('inf')
            }
        else:
            return {
                'temporal_scale': None,
                'spatial_scale': None,
                'characteristic_speed': None,
                'spatio_temporal_ratio': None
            }

    def spacetime_triangulation(self, temporal_data, spatial_data, time, spatial_coords):
        """
        Perform complete spacetime triangulation using both temporal and spatial data.
        """
        # Temporal triangulation
        temporal_results = self.temporal_analyzer.full_analysis(**temporal_data, time=time)
        
        # Spatial triangulation
        spatial_results = self.spatial_analyzer.spatial_triangulation(**spatial_data, coordinates_list=spatial_coords)
        
        # Combined analysis
        combined_analysis = self._combine_spacetime_analyses(temporal_results, spatial_results)
        
        results = {
            'temporal_triangulation': temporal_results,
            'spatial_triangulation': spatial_results,
            'spacetime_synthesis': combined_analysis
        }
        
        self.analysis_results['spacetime_triangulation'] = results
        return results

    def _combine_spacetime_analyses(self, temporal, spatial):
        """Synthesize temporal and spatial analyses into unified spacetime characterization."""
        tau_consensus = temporal.get('tau_consensus', 0)
        
        if 'spatial_triangulation' in spatial and 'agreement' in spatial['spatial_triangulation']:
            spatial_center = spatial['spatial_triangulation']['agreement']['mean_centroid']
            spatial_spread = spatial['spatial_triangulation']['agreement']['mean_spread']
        else:
            spatial_center = 0
            spatial_spread = 1
        
        # Spacetime volume metric
        spacetime_volume = tau_consensus * spatial_spread
        
        # Causality cone analysis
        if tau_consensus > 0 and spatial_spread > 0:
            causality_slope = spatial_spread / tau_consensus
        else:
            causality_slope = 0
        
        return {
            'spacetime_volume': float(spacetime_volume),
            'causality_slope': float(causality_slope),
            'effective_dimensionality': self._compute_effective_dimensionality(temporal, spatial),
            'memory_propagation': f"{causality_slope:.3f} spatial_units/time_unit"
        }

    def _compute_effective_dimensionality(self, temporal, spatial):
        """Compute effective dimensionality of spatio-temporal memory."""
        temporal_complexity = 'high' if temporal.get('system_type') in ['COMPLEX', 'MULTIPROCESS'] else 'low'
        
        spatial_complexity = 'low'
        if 'spatial_triangulation' in spatial:
            for key, result in spatial['spatial_triangulation'].items():
                if key.startswith('measurement_'):
                    if result.get('modality', {}).get('modality_type') in ['BIPOLAR', 'MULTIPLE_HOTSPOTS']:
                        spatial_complexity = 'high'
                        break
        
        if temporal_complexity == 'high' and spatial_complexity == 'high':
            return "HIGH_DIMENSIONAL"
        elif temporal_complexity == 'high' or spatial_complexity == 'high':
            return "MEDIUM_DIMENSIONAL"
        else:
            return "LOW_DIMENSIONAL"


            
#..........................................................................................................................................................

class QUMemoryAnalyzer:
    """
    Universal memory analyzer for cross-domain moment analysis.
    Supports: Environmental memory, statistical distributions, image analysis, 
    financial returns, physical densities, and any influence density function.
    """
    
    def __init__(self, sampling_rate: float = 1.0, rtol: float = 1e-6):
        self.sampling_rate = sampling_rate
        self.rtol = rtol
        self.supported_domains = [
            'time', 'space', 'velocity', 'statistics', 'image', 
            'finance', 'quantum', 'general'
        ]
    
    def from_influence_density(self, 
                             density: np.ndarray,
                             coordinate: np.ndarray, 
                             domain: str = "general",
                             signed: bool = False,
                             normalize: bool = True) -> Dict:
        """
        UNIVERSAL MOMENT ANALYZER - Core method for all domains
        
        Parameters:
        -----------
        density : array-like
            Influence density f(x) (e.g., |G(t)|, p(x), I(x,y), ρ(r), etc.)
        coordinate : array-like  
            Independent variable x (time, space, velocity, returns, etc.)
        domain : str
            Domain for interpretation: 'time', 'space', 'statistics', 'image', 'finance', etc.
        signed : bool
            Whether to use signed density (False = use absolute value)
        normalize : bool
            Whether to normalize to unit integral
        
        Returns:
        --------
        results : dict
            Complete moment analysis with domain-specific interpretation
        """
        f = np.asarray(density, dtype=np.float64)
        x = np.asarray(coordinate, dtype=np.float64)
        
        if f.shape != x.shape:
            raise ValueError("Density and coordinate must have same shape")
        
        # Handle signed densities
        if signed:
            f_unsigned = f
            sign_preserving = True
        else:
            f_unsigned = np.abs(f)
            sign_preserving = False
        
        # Remove zeros to avoid numerical issues
        mask = f_unsigned > self.rtol * np.max(f_unsigned)
        if np.sum(mask) == 0:
            raise ValueError("Density is effectively zero everywhere")
        
        f_clean = f_unsigned[mask]
        x_clean = x[mask]
        
        # Compute moments
        M0 = trapezoid(f_clean, x_clean)
        
        if normalize:
            mu = f_clean / M0
            total_influence = M0
        else:
            mu = f_clean
            total_influence = M0
        
        # First moment (centroid)
        M1 = trapezoid(x_clean * mu, x_clean)
        centroid = M1
        
        # Second moment and variance
        M2 = trapezoid(x_clean**2 * mu, x_clean)
        variance = M2 - centroid**2 if M2 > centroid**2 else 0.0
        
        # Higher moments (standardized)
        std = np.sqrt(variance) if variance > 1e-12 else 1.0
        
        if std > 1e-12:
            M3 = trapezoid(((x_clean - centroid) / std)**3 * mu, x_clean)
            M4 = trapezoid(((x_clean - centroid) / std)**4 * mu, x_clean)
            excess_kurtosis = M4 - 3
        else:
            M3 = 0.0
            excess_kurtosis = -3.0  # Minimum kurtosis for degenerate distribution
        
        # Modality analysis
        modality = self._analyze_modality(mu, x_clean)
        
        # Domain-specific interpretation
        interpretation = self._domain_interpretation(domain, centroid, variance, 
                                                   M3, excess_kurtosis, total_influence)
        
        return {
            'domain': domain,
            'centroid': float(centroid),
            'variance': float(variance),
            'std_dev': float(std),
            'skewness': float(M3),
            'kurtosis': float(excess_kurtosis),
            'total_influence': float(total_influence),
            'modality': modality,
            'distribution': mu,
            'coordinate': x_clean,
            'interpretation': interpretation,
            'sign_preserving': sign_preserving
        }
    
    def _domain_interpretation(self, domain: str, centroid: float, variance: float,
                             skewness: float, kurtosis: float, total_influence: float) -> Dict:
        """Generate domain-specific interpretation of moments"""
        base_interpretation = {
            'zeroth_moment': f"Total influence: {total_influence:.4f}",
            'first_moment': f"Centroid: {centroid:.4f}",
            'second_moment': f"Spread: {np.sqrt(variance):.4f}",
            'shape': f"Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}"
        }
        
        domain_specific = {}
        if domain == 'time':
            domain_specific = {
                'meaning': f"Temporal persistence: {centroid:.4f} units",
                'duration': f"Memory duration (std): {np.sqrt(variance):.4f}",
                'temporal_centroid': centroid
            }
        elif domain == 'space':
            domain_specific = {
                'meaning': f"Spatial centroid: {centroid:.4f} units",
                'spread': f"Spatial extent: {np.sqrt(variance):.4f}",
                'center_of_mass': centroid
            }
        elif domain == 'statistics':
            domain_specific = {
                'meaning': f"Mean: {centroid:.4f}",
                'spread': f"Standard deviation: {np.sqrt(variance):.4f}",
                'distribution_shape': f"Skewed {'right' if skewness > 0 else 'left'}, "
                                    f"{'heavy-tailed' if kurtosis > 0 else 'light-tailed'}"
            }
        elif domain == 'finance':
            domain_specific = {
                'meaning': f"Expected return: {centroid:.4f}",
                'risk': f"Volatility: {np.sqrt(variance):.4f}",
                'return_characteristics': f"Asymmetric: {skewness:.4f}, "
                                        f"Tail risk: {kurtosis:.4f}"
            }
        elif domain == 'image':
            domain_specific = {
                'meaning': f"Intensity centroid: {centroid:.4f}",
                'spatial_extent': f"Region size: {np.sqrt(variance):.4f}",
                'texture': f"Shape asymmetry: {skewness:.4f}"
            }
        
        return {**base_interpretation, **domain_specific}
    
    def _analyze_modality(self, density: np.ndarray, coordinate: np.ndarray) -> Dict:
        """Analyze modality of distribution (peaks, multimodality)"""
        from scipy.signal import find_peaks
        
        # Simple peak detection
        peaks, properties = find_peaks(density, height=np.max(density)*0.1, distance=len(density)//10)
        
        modality_info = {
            'n_peaks': len(peaks),
            'peak_locations': coordinate[peaks].tolist() if len(peaks) > 0 else [],
            'peak_heights': density[peaks].tolist() if len(peaks) > 0 else [],
            'modality_type': 'multimodal' if len(peaks) > 1 else 'unimodal'
        }
        
        return modality_info
    
    # ORIGINAL MEMORY ANALYSIS METHODS (compatible with QTMemoryAnalyzer)
    
    def from_impulse_response(self, G: np.ndarray, t: np.ndarray, 
                            method: str = 'unsigned') -> Dict:
        """
        Environmental memory analysis from impulse response
        """
        if method == 'unsigned':
            density = np.abs(G)
            signed = False
        elif method == 'signed':
            density = G
            signed = True
        else:
            raise ValueError("method must be 'unsigned' or 'signed'")
        
        return self.from_influence_density(density, t, domain='time', signed=signed)
    
    def from_step_response(self, S: np.ndarray, t: np.ndarray) -> Dict:
        """
        Step response memory analysis
        """
        # Compute deviation from final value
        S_final = S[-1] if len(S) > 0 else 0
        if np.abs(S_final) < self.rtol:
            S_final = 1.0  # Avoid division by zero
        
        deviation = 1 - S / S_final
        density = np.abs(deviation)
        
        return self.from_influence_density(density, t, domain='time')
    
    def from_time_series(self, X: np.ndarray, t: np.ndarray) -> Dict:
        """
        Autocorrelation-based memory analysis
        """
        # Compute autocorrelation
        X_centered = X - np.mean(X)
        autocorr = correlate(X_centered, X_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Use first half to avoid noise
        n_half = len(autocorr) // 2
        autocorr_half = autocorr[:n_half]
        t_half = t[:n_half]
        
        density = np.abs(autocorr_half)
        
        return self.from_influence_density(density, t_half, domain='time')
    
    def triangulate_memory(self, G: np.ndarray, S: np.ndarray, 
                          X: np.ndarray, t: np.ndarray) -> Dict:
        """
        Triangulation protocol using all three methods
        """
        results = {}
        
        # Method 1: Impulse response
        if G is not None:
            results['impulse_method'] = self.from_impulse_response(G, t)
        
        # Method 2: Step response  
        if S is not None:
            results['step_method'] = self.from_step_response(S, t)
        
        # Method 3: Time series autocorrelation
        if X is not None:
            results['autocorr_method'] = self.from_time_series(X, t)
        
        # Compute agreement metrics
        if len(results) > 1:
            centroids = [r['centroid'] for r in results.values()]
            results['agreement'] = {
                'mean_centroid': np.mean(centroids),
                'std_centroid': np.std(centroids),
                'max_disagreement': np.max(centroids) - np.min(centroids),
                'agreement_quality': 'high' if np.std(centroids) < 0.1 else 'low'
            }
        
        return results
    
    # NEW CROSS-DOMAIN METHODS
    
    def from_probability_distribution(self, samples: np.ndarray, 
                                    bins: int = 100, 
                                    domain: str = 'statistics') -> Dict:
        """Analyze statistical probability distributions"""
        hist, bin_edges = np.histogram(samples, bins=bins, density=True)
        x = (bin_edges[:-1] + bin_edges[1:]) / 2
        return self.from_influence_density(hist, x, domain=domain)
    
    def from_image_moments(self, image: np.ndarray, 
                          axis: int = 0,
                          domain: str = 'image') -> Dict:
        """Analyze image intensity distributions along specified axis"""
        profile = np.sum(image, axis=axis)
        coords = np.arange(len(profile))
        return self.from_influence_density(profile, coords, domain=domain)
    
    def from_spatial_field(self, field: np.ndarray, 
                          coordinates: Tuple[np.ndarray, ...],
                          domain: str = 'space') -> Dict:
        """Analyze spatial fields (1D, 2D, 3D)"""
        if len(coordinates) == 1:
            # 1D field
            return self.from_influence_density(np.abs(field), coordinates[0], domain=domain)
        else:
            # Multi-dimensional - analyze marginal distributions
            results = {}
            for i, coord in enumerate(coordinates):
                # Sum over other dimensions
                sum_axes = tuple(j for j in range(len(coordinates)) if j != i)
                marginal = np.sum(np.abs(field), axis=sum_axes)
                results[f'axis_{i}'] = self.from_influence_density(marginal, coord, domain=domain)
            return results
    
    def compare_domains(self, densities: List[np.ndarray], 
                       coordinates: List[np.ndarray],
                       domains: List[str]) -> Dict:
        """Compare moment analyses across different domains"""
        if len(densities) != len(coordinates) or len(densities) != len(domains):
            raise ValueError("All input lists must have same length")
        
        comparisons = {}
        for i, (dens, coord, domain) in enumerate(zip(densities, coordinates, domains)):
            results = self.from_influence_density(dens, coord, domain=domain)
            comparisons[f'{domain}_{i}'] = results
        
        # Cross-domain analysis
        centroids = [r['centroid'] for r in comparisons.values()]
        spreads = [r['std_dev'] for r in comparisons.values()]
        
        comparisons['cross_domain'] = {
            'centroid_range': np.ptp(centroids),
            'spread_range': np.ptp(spreads),
            'relative_scales': {name: f"{r['centroid']:.3f} ± {r['std_dev']:.3f}" 
                              for name, r in comparisons.items()}
        }
        
        return comparisons
    
    def get_moment_interpretation_guide(self) -> Dict:
        """Return universal interpretation guide for moments"""
        return {
            'zeroth_moment': "Total quantity or influence",
            'first_moment': "Center, mean, or centroid location", 
            'second_moment': "Spread, variance, or extent",
            'variance': "Dispersion around center",
            'skewness': "Asymmetry (positive = right-tailed, negative = left-tailed)",
            'kurtosis': "Peakedness and tail behavior (positive = heavy-tailed, negative = light-tailed)",
            'universal_pattern': "0: How much? 1: Where? 2: How wide? 3: How asymmetric? 4: How peaked?"
        }


