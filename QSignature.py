"""
QSignature: Clean, organized causal persistence timescale estimators.
Version: 1.0 - Complete with QSpace3d, Q_all, Q_one, and extended diagnostics.
"""

import numpy as np
from scipy.signal import hilbert, welch
from typing import Optional, Literal, Dict, Tuple, Any, List
from enum import Enum
import warnings
import pandas as pd

# ============================================================================
# CORE ESTIMATORS (8 essential functions)
# ============================================================================

def tau_g(t: np.ndarray, R: np.ndarray, R_inf: Optional[float] = None) -> float:
    """τ_g: Generalized memory persistence timescale."""
    t = np.asarray(t, dtype=float)
    R = np.asarray(R, dtype=float)
    
    if R_inf is None:
        R_inf = R[-1]
    
    deviation = np.abs(R - R_inf)
    
    if np.all(deviation == 0) or np.sum(deviation) == 0:
        return 0.0
    
    numerator = np.trapezoid(t * deviation, t)
    denominator = np.trapezoid(deviation, t)
    
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    
    return numerator / denominator


def tau_s(t: np.ndarray, R: np.ndarray, R_inf: Optional[float] = None, 
          eps: float = 1e-6, _depth: int = 0, _max_depth: int = 5) -> float:
    """τ_s: Signed centroid timescale."""
    t = np.asarray(t, dtype=float)
    R = np.asarray(R, dtype=float)
    
    if _depth >= _max_depth:
        return 0.0
    
    R0 = R[0]
    if R_inf is None:
        R_inf = estimate_Rinf(t, R, method='auto')
    
    step_height = R_inf - R0
    
    if abs(step_height) < eps:
        R_centered = R - np.mean(R)
        if np.ptp(R_centered) < eps:
            return 0.0
        return tau_s(t, R_centered, R_inf=0.0, eps=eps, _depth=_depth + 1, _max_depth=_max_depth)
    
    dt = np.mean(np.diff(t))
    dR = np.gradient(R, dt)
    
    integrand = t * dR
    integral = np.trapezoid(integrand, t)
    
    return integral / step_height


def tau_2(t: np.ndarray, R: np.ndarray, R_inf: Optional[float] = None) -> float:
    """τ₂: Step-response timescale."""
    t = np.asarray(t, dtype=float)
    R = np.asarray(R, dtype=float)
    
    R0 = R[0]
    if R_inf is None:
        R_inf = estimate_Rinf(t, R, method='auto')
    
    step_height = R_inf - R0
    if abs(step_height) < 1e-6:
        return 0.0
    
    R_norm = (R - R0) / step_height
    f = 1 - R_norm
    return np.trapezoid(f, t)


def tau_u(t: np.ndarray, R: np.ndarray, R_inf: Optional[float] = None) -> float:
    """τ_u: Unsigned centroid timescale."""
    t = np.asarray(t, dtype=float)
    R = np.asarray(R, dtype=float)
    
    if R_inf is None:
        R_inf = R[-1]
    
    dt = np.mean(np.diff(t))
    dR = np.gradient(R, dt)
    
    numerator = np.trapezoid(t * np.abs(dR), t)
    denominator = np.trapezoid(np.abs(dR), t)
    
    return numerator / denominator if denominator != 0 else np.nan


def tau_3(t: np.ndarray, R: np.ndarray, 
          method: Literal['autocorrelation', 'impulse', 'autocorr_centered'] = 'autocorrelation',
          R_inf: Optional[float] = None) -> float:
    """τ₃: Timescale estimator via autocorrelation."""
    t = np.asarray(t, dtype=float)
    
    if method in ['autocorrelation', 'autocorr_centered']:
        R = np.asarray(R, dtype=float)
        dt = t[1] - t[0]
        
        if method == 'autocorr_centered':
            if R_inf is None:
                R_inf = R[-1]
            R_centered = R - R_inf
        else:
            R_centered = R - np.mean(R)
        
        n = len(R_centered)
        
        autocorr = np.correlate(R_centered, R_centered, mode='full')[n-1:]
        autocorr = autocorr / (n - np.arange(n))
        autocorr = autocorr / autocorr[0]
        
        idx = np.where(autocorr < 0.01)[0]
        cutoff = idx[0] if len(idx) > 0 else len(autocorr)
        
        return np.trapezoid(autocorr[:cutoff], dx=dt)
    
    elif method == 'impulse':
        g = np.asarray(R, dtype=float)
        integral_g2 = np.trapezoid(g**2, t)
        if integral_g2 == 0:
            return np.nan
        return 1.0 / (2.0 * integral_g2)
    
    else:
        raise ValueError(f"Method must be 'autocorrelation', 'autocorr_centered', or 'impulse'")


def tau_pole(t: np.ndarray, R: np.ndarray, R_inf: Optional[float] = None) -> float:
    """τ_pole: Spectral pole timescale."""
    t = np.asarray(t, dtype=float)
    R = np.asarray(R, dtype=float)
    
    if R_inf is None:
        R_inf = estimate_Rinf(t, R, method='auto')
    
    dt = np.mean(np.diff(t))
    dR = np.gradient(R, dt)
    
    R_tilde_0 = R_inf
    R_tilde_prime_0 = -np.trapezoid(t * dR, t)
    
    if abs(R_tilde_0) < 1e-10:
        return np.nan
    
    return np.abs(R_tilde_prime_0 / R_tilde_0)


def tau_env(t: np.ndarray, R: np.ndarray, method: Literal['hilbert', 'peak'] = 'hilbert') -> float:
    """τ_env: Envelope decay timescale."""
    t = np.asarray(t, dtype=float)
    R = np.asarray(R, dtype=float)
    
    if method == 'hilbert':
        dt = t[1] - t[0]
        dR = np.gradient(R, dt)
        
        analytic = hilbert(dR)
        envelope = np.abs(analytic)
        
        E0 = envelope[0] if envelope[0] > 0 else np.max(envelope)
        idx = np.where(envelope/E0 < 0.05)[0]
        cutoff = idx[0] if len(idx) > 0 else len(t)
        
        return np.trapezoid(envelope[:cutoff]/E0, t[:cutoff])
    
    elif method == 'peak':
        peaks = []
        for i in range(1, len(R)-1):
            if abs(R[i]) > abs(R[i-1]) and abs(R[i]) > abs(R[i+1]):
                peaks.append((t[i], abs(R[i])))
        
        if len(peaks) < 3:
            return np.nan
        
        t_peaks = np.array([p[0] for p in peaks])
        R_peaks = np.array([p[1] for p in peaks])
        coeff = np.polyfit(t_peaks, np.log(R_peaks), 1)
        alpha = -coeff[0]
        
        return 1/alpha if alpha > 0 else np.nan
    
    else:
        raise ValueError(f"Method must be 'hilbert' or 'peak'")


def tau_E(t: np.ndarray, R: np.ndarray, method: Literal['autocorrelation', 'peak'] = 'autocorrelation',
          omega0: Optional[float] = None) -> float:
    """τ_E: Energy decay timescale."""
    t = np.asarray(t, dtype=float)
    R = np.asarray(R, dtype=float)
    dt = t[1] - t[0]
    
    if method == 'autocorrelation':
        dR = np.gradient(R, dt)
        
        if omega0 is None:
            freqs, psd = welch(R, fs=1/dt, nperseg=min(1024, len(R)))
            omega0 = 2*np.pi*freqs[np.argmax(psd[1:]) + 1]
        
        E = 0.5*dR**2 + 0.5*omega0**2*R**2
        E_centered = E - np.mean(E)
        n = len(E_centered)
        
        autocorr = np.correlate(E_centered, E_centered, mode='full')[n-1:]
        autocorr = autocorr / (n - np.arange(n))
        autocorr = autocorr / autocorr[0]
        
        idx = np.where(autocorr < 0.05)[0]
        cutoff = idx[0] if len(idx) > 0 else n//2
        
        return np.trapezoid(autocorr[:cutoff], dx=dt)
    
    elif method == 'peak':
        dR = np.gradient(R, dt)
        
        peaks = []
        for i in range(1, len(R)-1):
            if R[i] > R[i-1] and R[i] > R[i+1]:
                peaks.append(t[i])
        
        if len(peaks) > 2:
            T = np.mean(np.diff(peaks[:3]))
            omega0 = 2*np.pi/T if T > 0 else 1.0
        else:
            omega0 = 1.0
        
        E = 0.5*dR**2 + 0.5*omega0**2*R**2
        
        E_peaks = []
        for i in range(1, len(E)-1):
            if E[i] > E[i-1] and E[i] > E[i+1]:
                E_peaks.append((t[i], E[i]))
        
        if len(E_peaks) < 3:
            return np.nan
        
        t_peaks = np.array([p[0] for p in E_peaks])
        E_vals = np.array([p[1] for p in E_peaks])
        
        coeff = np.polyfit(t_peaks, np.log(E_vals), 1, w=np.sqrt(E_vals))
        decay_rate = -coeff[0]
        
        return 1/decay_rate if decay_rate > 0 else np.nan
    
    else:
        raise ValueError(f"Method must be 'autocorrelation' or 'peak'")


# ============================================================================
# DIAGNOSTIC RATIOS
# ============================================================================

def Delta_su(t: np.ndarray, R: np.ndarray, R_inf: Optional[float] = None,
             ensure_proper_window: bool = True, verbose: bool = False) -> float:
    """Δₛᵤ = (τ_s - τ_u)/τ_u - Oscillation strength."""
    tau_s_val = tau_s(t, R, R_inf)
    tau_u_val = tau_u(t, R, R_inf)
    
    if tau_u_val == 0 or not np.isfinite(tau_s_val) or not np.isfinite(tau_u_val):
        return np.nan
    
    if ensure_proper_window and tau_s_val < 0 and verbose:
        print(f"Warning: τ_s = {tau_s_val:.3f} < 0. Time window may be too short.")
    
    return (tau_s_val - tau_u_val) / tau_u_val


def Delta_23_env(t: np.ndarray, R: np.ndarray, method: Literal['hilbert', 'peak'] = 'hilbert') -> float:
    """Δ₂₃ᵉⁿᵛ = (τ_env - τ_E)/τ_env - Damping linearity."""
    tau_env_val = tau_env(t, R, method=method)
    
    if method == 'hilbert':
        tau_E_val = tau_E(t, R, method='autocorrelation')
    else:
        tau_E_val = tau_E(t, R, method='peak')
    
    if tau_env_val == 0 or not np.isfinite(tau_env_val) or not np.isfinite(tau_E_val):
        return np.nan
    
    return np.clip((tau_env_val - tau_E_val) / tau_env_val, -1, 1)


def Delta_ps(t: np.ndarray, R: np.ndarray, method: Literal['hilbert', 'peak'] = 'hilbert') -> float:
    """Δ_ps: Pole-envelope discrepancy ratio."""
    tau_env_val = tau_env(t, R, method=method)
    tau_pole_val = tau_pole(t, R)
    
    if np.isnan(tau_env_val) or np.isnan(tau_pole_val):
        return np.nan
    if abs(tau_env_val + tau_pole_val) < 1e-10:
        return np.nan
    
    return (tau_env_val - tau_pole_val) / (tau_env_val + tau_pole_val)


def rho_13_step(t: np.ndarray, R: np.ndarray, R_inf: Optional[float] = None) -> float:
    """ρ₁₃ = τ_s/τ₃ - Memory complexity."""
    tau_s_val = tau_s(t, R, R_inf)
    tau_3_val = tau_3(t, R, method='autocorr_centered', R_inf=R_inf)
    
    if tau_3_val == 0 or not np.isfinite(tau_s_val) or not np.isfinite(tau_3_val):
        return np.nan
    
    return tau_s_val / tau_3_val


def rho_13_impulse(t: np.ndarray, g: np.ndarray) -> float:
    """ρ₁₃ for impulse responses."""
    if np.all(g == 0):
        return np.nan
    
    H = np.trapezoid(g, t)
    if H == 0:
        return np.nan
    
    tau_gs = np.trapezoid(t * g, t) / H
    tau_3_val = tau_3(t, g, method='impulse')
    
    if tau_3_val == 0 or not np.isfinite(tau_gs) or not np.isfinite(tau_3_val):
        return np.nan
    
    return tau_gs / tau_3_val


# ============================================================================
# MAIN COMPUTATION FUNCTION
# ============================================================================

def compute_all(t: np.ndarray, R: np.ndarray, R_inf: Optional[float] = None,
                g: Optional[np.ndarray] = None,
                method: Literal['practical', 'theorem', 'both'] = 'practical',
                verbose: bool = False) -> Dict[str, float]:
    """Compute all timescale estimators and diagnostic ratios."""
    if R_inf is None:
        R_inf = estimate_Rinf(t, R, method='auto')
    
    results = {}
    
    # Core estimators
    results['tau_g'] = tau_g(t, R, R_inf)
    results['tau_s'] = tau_s(t, R, R_inf)
    results['tau_u'] = tau_u(t, R, R_inf)
    results['tau_2'] = tau_2(t, R, R_inf)
    results['tau_3'] = tau_3(t, R, method='autocorr_centered', R_inf=R_inf)
    results['tau_pole'] = tau_pole(t, R, R_inf)
    
    # Basic diagnostics
    results['Delta_su'] = Delta_su(t, R, R_inf, ensure_proper_window=True, verbose=verbose)
    results['rho_13_step'] = rho_13_step(t, R, R_inf)
    
    # Envelope/Energy methods
    if method in ['practical', 'both']:
        results['tau_env_practical'] = tau_env(t, R, method='hilbert')
        results['tau_E_practical'] = tau_E(t, R, method='autocorrelation')
        results['Delta_23_env_practical'] = Delta_23_env(t, R, method='hilbert')
        results['Delta_ps_practical'] = Delta_ps(t, R, method='hilbert')
    
    if method in ['theorem', 'both']:
        results['tau_env_theorem'] = tau_env(t, R, method='peak')
        results['tau_E_theorem'] = tau_E(t, R, method='peak')
        results['Delta_23_env_theorem'] = Delta_23_env(t, R, method='peak')
        results['Delta_ps_theorem'] = Delta_ps(t, R, method='peak')
    
    # Theorem 3.4 specific
    if g is not None:
        H = np.trapezoid(g, t)
        if H != 0:
            results['tau_g_s'] = np.trapezoid(t * g, t) / H
            results['tau_3_impulse'] = tau_3(t, g, method='impulse')
            results['rho_13_impulse'] = rho_13_impulse(t, g)
            results['H_normalization'] = H
    
    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def estimate_Rinf(t: np.ndarray, R: np.ndarray, method: str = 'auto') -> float:
    """Robust estimation of R_infinity (steady-state value)."""
    t = np.asarray(t, dtype=float)
    R = np.asarray(R, dtype=float)
    
    if method == 'integral':
        dt = np.mean(np.diff(t))
        dR = np.gradient(R, dt)
        return R[0] + np.trapezoid(dR, t)
    
    elif method == 'tail':
        tail_len = max(5, len(R) // 10)
        return np.mean(R[-tail_len:])
    
    elif method == 'last':
        return R[-1]
    
    elif method == 'auto':
        dt = np.mean(np.diff(t))
        dR = np.gradient(R, dt)
        sign_changes = np.sum((dR[:-1] * dR[1:]) < 0)
        
        if sign_changes >= 3:
            if abs(R[-1] - R[0]) > 0.1 * np.ptp(R):
                return R[-1]
            return estimate_Rinf(t, R, method='tail')
        else:
            return R[-1]
    
    else:
        raise ValueError("method must be 'integral', 'tail', 'last', or 'auto'")


def print_summary(results: Dict[str, float], title: str = "τ Computation Results") -> None:
    """Print formatted summary of tau computation results."""
    print("\n" + "="*60)
    print(title)
    print("="*60)
    
    categories = {
        'Core Timescales': ['tau_g', 'tau_s', 'tau_u', 'tau_2', 'tau_3', 'tau_pole'],
        'Envelope & Energy': ['tau_env_practical', 'tau_E_practical', 
                              'tau_env_theorem', 'tau_E_theorem'],
        'Diagnostic Ratios': ['Delta_su', 'Delta_23_env_practical', 'Delta_ps_practical',
                              'Delta_23_env_theorem', 'Delta_ps_theorem', 
                              'rho_13_step', 'rho_13_impulse'],
        'Theorem 3.4': ['tau_g_s', 'tau_3_impulse', 'H_normalization']
    }
    
    for category, keys in categories.items():
        print(f"\n📊 {category}:")
        printed = False
        for key in keys:
            if key in results and results[key] is not None and np.isfinite(results[key]):
                print(f"  {key:25} = {results[key]:.6f}")
                printed = True
        if not printed:
            print(f"  (No {category.lower()} available)")
    
    print("\n" + "="*60)


def validate_theorems(t: np.ndarray, R: np.ndarray, R_inf: Optional[float] = None,
                      g: Optional[np.ndarray] = None) -> Dict[str, Dict[str, any]]:
    """Validate all theorems from the paper."""
    results = compute_all(t, R, R_inf, g, method='both', verbose=False)
    
    validation = {}
    
    validation['Theorem 3.1'] = {
        'tau_s': results.get('tau_s', np.nan),
        'tau_2': results.get('tau_2', np.nan),
        'difference': abs(results.get('tau_s', 0) - results.get('tau_2', 0)),
        'relative_error': abs(results.get('tau_s', 1) - results.get('tau_2', 1)) / 
                         (results.get('tau_s', 1) + 1e-10) * 100,
        'valid': abs(results.get('tau_s', 0) - results.get('tau_2', 0)) < 1e-4
    }
    
    validation['Theorem 3.3'] = {
        'tau_s': results.get('tau_s', np.nan),
        'tau_pole': results.get('tau_pole', np.nan),
        'difference': abs(results.get('tau_s', 0) - results.get('tau_pole', 0)),
        'relative_error': abs(results.get('tau_s', 1) - results.get('tau_pole', 1)) / 
                         (results.get('tau_s', 1) + 1e-10) * 100,
        'valid': abs(results.get('tau_s', 0) - results.get('tau_pole', 0)) < 1e-4
    }
    
    if g is not None and 'rho_13_impulse' in results:
        rho = results['rho_13_impulse']
        validation['Theorem 3.4'] = {
            'rho': rho,
            'is_exponential': abs(rho - 1) < 0.1,
            'is_non_exponential': rho > 1.5,
            'valid': np.isfinite(rho)
        }
    
    delta_su = results.get('Delta_su', np.nan)
    validation['Theorem 3.6'] = {
        'Delta_su': delta_su,
        'is_oscillatory': delta_su < -0.8,
        'is_monotonic': abs(delta_su) < 0.2,
        'valid': np.isfinite(delta_su)
    }
    
    return validation


# ============================================================================
# QSPACE3D AND RELATED FUNCTIONS
# ============================================================================

class responseType(Enum):
    STEP_RESPONSE = "step_response"
    IMPULSE_RESPONSE = "impulse_response"
    STATIONARY = "stationary"
    TRENDING = "trending"
    OSCILLATORY = "oscillatory"
    ARBITRARY = "arbitrary"
    INVALID = "invalid"


def validate_input(t: np.ndarray, R: np.ndarray, min_points: int = 10):
    """Validate input arrays for QSpace3d."""
    t = np.asarray(t, dtype=float)
    R = np.asarray(R, dtype=float)
    
    if len(t) != len(R):
        return False, f"t and R must have same length (got {len(t)} vs {len(R)})"
    
    if len(t) < min_points:
        return False, f"Need at least {min_points} points (got {len(t)})"
    
    if np.all(R == R[0]):
        return False, "response is constant — no dynamics to analyze"
    
    if np.any(np.isnan(t)) or np.any(np.isnan(R)):
        return False, "NaN values detected — please clean data first"
    
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        return False, "t must be strictly increasing"
    
    return True, ""


def normalize_response(R: np.ndarray, method: str = 'minmax'):
    """Normalize response to canonical scale."""
    R = np.asarray(R, dtype=float)
    original_min = np.min(R)
    original_max = np.max(R)
    original_mean = np.mean(R)
    original_std = np.std(R)
    
    if method == 'minmax':
        if original_max - original_min > 1e-10:
            R_norm = (R - original_min) / (original_max - original_min)
        else:
            R_norm = R - original_mean
    
    elif method == 'zscore':
        if original_std > 1e-10:
            R_norm = (R - original_mean) / original_std
        else:
            R_norm = R - original_mean
    
    elif method == 'unit_area':
        area = np.trapezoid(np.abs(R - np.mean(R)), dx=1.0)
        if area > 1e-10:
            R_norm = (R - np.mean(R)) / area
        else:
            R_norm = R - np.mean(R)
    
    else:
        R_norm = R.copy()
    
    return R_norm, {
        'original_min': original_min,
        'original_max': original_max,
        'original_mean': original_mean,
        'original_std': original_std,
        'method': method
    }


def detect_response_type(t: np.ndarray, R: np.ndarray):
    """Detect the type of response for metadata purposes."""
    t = np.asarray(t, dtype=float)
    R = np.asarray(R, dtype=float)
    dt = np.mean(np.diff(t)) if len(t) > 1 else 1.0
    n = len(R)
    
    R0 = R[0]
    R_end = R[-1]
    R_range = np.ptp(R)
    
    is_step_like = abs(R_end - R0) > 0.5 * R_range if R_range > 1e-10 else False
    
    baseline = np.mean(R[:min(10, n)])
    returns_to_baseline = abs(R[-1] - baseline) < 0.1 * np.std(R) if np.std(R) > 1e-10 else False
    
    is_stationary = False
    try:
        from scipy.stats import linregress
        if n > 2 and np.std(R) > 1e-10:
            slope, _, _, _, _ = linregress(t, R)
            is_stationary = abs(slope) < 0.01 * np.std(R) / (t[-1] - t[0])
    except:
        pass
    
    dR = np.gradient(R, dt)
    zero_crossings = np.sum((dR[:-1] * dR[1:]) < 0)
    is_oscillatory = zero_crossings > 5
    
    if is_step_like and not is_oscillatory:
        response_type = responseType.STEP_RESPONSE
    elif returns_to_baseline:
        response_type = responseType.IMPULSE_RESPONSE
    elif is_stationary and is_oscillatory:
        response_type = responseType.STATIONARY
    elif not is_stationary and not returns_to_baseline:
        response_type = responseType.TRENDING
    elif is_oscillatory and not returns_to_baseline:
        response_type = responseType.OSCILLATORY
    else:
        response_type = responseType.ARBITRARY
    
    return {
        'type': response_type,
        'is_step_like': is_step_like,
        'returns_to_baseline': returns_to_baseline,
        'is_stationary': is_stationary,
        'is_oscillatory': is_oscillatory,
        'zero_crossings': zero_crossings
    }


# ============================================================================
# QSPACE3D HELPER FUNCTIONS
# ============================================================================

def _compute_confidence_metrics(t, R, tau_s, tau_u, tau_2, tau_3, Z, is_oscillatory):
    """Compute confidence metrics for QSpace3d signature."""
    t_max = t[-1] - t[0]
    
    window_ratio = t_max / tau_2 if tau_2 and not np.isnan(tau_2) and tau_2 > 0 else 0
    window_adequate = window_ratio > 5
    
    tau_vals = [tau_s, tau_u, tau_2, tau_3]
    tau_vals = [v for v in tau_vals if v is not None and not np.isnan(v) and v > 0]
    if len(tau_vals) >= 3:
        tau_consistency = np.std(tau_vals) / np.mean(tau_vals)
        estimators_agree = tau_consistency < 0.3
    else:
        tau_consistency = np.nan
        estimators_agree = False
    
    tau_3_stable = tau_3 is not None and not np.isnan(tau_3) and tau_3 > 0.01 * t_max
    Y_reliable = tau_3_stable
    
    Z_meaningful = False
    if is_oscillatory:
        Z_meaningful = Z is not None and not np.isnan(Z) and -1 <= Z <= 1
    else:
        Z_meaningful = True
    
    quality_score = 1.0
    if not window_adequate:
        quality_score *= 0.5
    if not estimators_agree:
        quality_score *= 0.8
    if not Y_reliable:
        quality_score *= 0.7
    if is_oscillatory and not Z_meaningful:
        quality_score *= 0.8
    
    quality_score = np.clip(quality_score, 0, 1)
    
    if quality_score > 0.8:
        quality_level = 'high'
    elif quality_score > 0.5:
        quality_level = 'medium'
    else:
        quality_level = 'low'
    
    return {
        'window_ratio': window_ratio,
        'window_adequate': window_adequate,
        'tau_consistency': tau_consistency,
        'estimators_agree': estimators_agree,
        'tau_3_stable': tau_3_stable,
        'Y_reliable': Y_reliable,
        'Z_meaningful': Z_meaningful,
        'quality_score': quality_score,
        'quality_level': quality_level
    }


def _validate_signature(X, Y, Z, tau_2, is_oscillatory):
    """Validate signature against physical bounds."""
    issues = []
    warnings_list = []
    is_plausible = True
    
    if X is not None and not np.isnan(X):
        if X < -3.5:
            issues.append(f"X={X:.2f} is too negative (possible numerical issue)")
            is_plausible = False
        if X > 2.0:
            issues.append(f"X={X:.2f} is too positive (unusual growth)")
            is_plausible = False
    
    if Y is not None and not np.isnan(Y):
        if abs(Y) < 0.01:
            warnings_list.append(f"Y≈0 suggests τ₃ is very large (check window length)")
        if Y > 10:
            warnings_list.append(f"Y={Y:.2f} is very large (τ₃ very small)")
    
    if is_oscillatory and Z is not None and not np.isnan(Z):
        if Z < -1 or Z > 1:
            issues.append(f"Z={Z:.2f} outside physical bounds [-1, 1]")
            is_plausible = False
    
    if tau_2 is not None and not np.isnan(tau_2):
        if tau_2 < 0:
            issues.append(f"τ₂={tau_2:.4f} is negative (window too short for response)")
            is_plausible = False
    
    return issues, warnings_list, is_plausible


def _detect_special_case(t, R):
    """Detect special cases: conservative oscillator, etc."""
    R_centered = R - np.mean(R)
    if np.std(R) == 0:
        return True, 'constant'
    
    n = len(R)
    seg1 = np.abs(R_centered[:n//3])
    seg2 = np.abs(R_centered[n//3:2*n//3])
    seg3 = np.abs(R_centered[2*n//3:])
    
    amp1 = np.max(seg1)
    amp2 = np.max(seg2)
    amp3 = np.max(seg3)
    
    is_constant_amplitude = (amp2 > 0.9 * amp1) and (amp3 > 0.9 * amp1)
    
    dt = np.mean(np.diff(t))
    dR = np.gradient(R, dt)
    zero_crossings = np.sum((dR[:-1] * dR[1:]) < 0)
    is_oscillatory = zero_crossings > 5
    
    is_centered = abs(np.mean(R)) < 0.05 * np.std(R)
    
    if is_constant_amplitude and is_oscillatory and is_centered:
        if abs(R[0] - np.max(R)) < 0.1 * np.std(R):
            return True, 'conservative_cosine'
        else:
            return True, 'conservative_sine'
    
    return False, None


def _apply_special_case(case, include_tau2):
    """Apply theoretical values for special cases."""
    if case == 'conservative_cosine':
        X, Y, Z = -3.0, 0.5, 1.0
        tau_2_val = np.inf
    elif case == 'conservative_sine':
        X, Y, Z = -1.0, 0.5, 1.0
        tau_2_val = np.inf
    elif case == 'constant':
        X, Y, Z = np.nan, np.nan, np.nan
        tau_2_val = np.nan
    else:
        return None
    
    output = {
        'X': X,
        'Y': Y,
        'Z': Z,
        'signature_3d': np.array([X, Y, Z]),
        'confidence': {
            'window_ratio': np.inf if case.startswith('conservative') else 0,
            'window_adequate': True,
            'tau_consistency': 0.0,
            'estimators_agree': True,
            'tau_3_stable': True,
            'Y_reliable': True,
            'Z_meaningful': True,
            'quality_score': 1.0,
            'quality_level': 'high'
        },
        'validation': {
            'issues': [],
            'warnings': [],
            'is_plausible': True,
            'needs_review': False
        },
        'method_used': {
            'tau_3_method': 'theoretical',
            'tau_env_method': 'theoretical',
            'tau_E_method': 'theoretical',
            'Rinf_method': 'special_case',
            'fallback_used': False,
            'special_case': case
        },
        'metadata': {
            'is_oscillatory': case.startswith('conservative'),
            'response_type': 'conservative_oscillator',
            'normalization_applied': False,
            'Rinf_used': 0.0,
            'Rinf_method': 'special_case',
            'compute_method': 'special_case',
            'n_points': 0,
            'duration': 0,
            'oscillation_threshold': -0.2,
            'conservative': True,
            'near_conservative': False,
            'special_case': case
        }
    }
    
    if include_tau2:
        output['tau_2'] = tau_2_val
    
    return output


def _compute_extended_diagnostics(t, R, results, tau_s, tau_u):
    """Compute extended diagnostics for advanced users."""
    extended = {}
    
    dt = np.mean(np.diff(t))
    dR = np.gradient(R, dt)
    
    # Skewness
    if np.sum(np.abs(dR)) > 0:
        t_centered = t - tau_s
        skew_num = np.trapezoid(t_centered**3 * np.abs(dR), t)
        skew_den = np.trapezoid(t_centered**2 * np.abs(dR), t)**1.5
        extended['skewness'] = skew_num / (skew_den + 1e-10)
    else:
        extended['skewness'] = 0.0
    
    # Kurtosis
    if np.sum(np.abs(dR)) > 0:
        kurt_num = np.trapezoid(t_centered**4 * np.abs(dR), t)
        kurt_den = np.trapezoid(t_centered**2 * np.abs(dR), t)**2
        extended['kurtosis'] = kurt_num / (kurt_den + 1e-10) - 3
    else:
        extended['kurtosis'] = 0.0
    
    # Zero-crossing rate
    zero_crossings = np.sum((dR[:-1] * dR[1:]) < 0)
    extended['zero_crossing_rate'] = zero_crossings / (t[-1] - t[0])
    
    # Spectral centroid
    if len(R) > 10:
        try:
            freqs, psd = welch(R, fs=1/dt, nperseg=min(256, len(R)//2))
            if np.sum(psd) > 0:
                extended['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
            else:
                extended['spectral_centroid'] = np.nan
        except:
            extended['spectral_centroid'] = np.nan
    else:
        extended['spectral_centroid'] = np.nan
    
    # Approximate entropy
    def _approx_entropy(x, m=2, r=0.2):
        if len(x) < m+1 or np.std(x) == 0:
            return 0
        N = len(x)
        r = r * np.std(x)
        def _phi(m):
            patterns = [x[i:i+m] for i in range(N-m+1)]
            C = []
            for pattern in patterns:
                count = sum(1 for p in patterns if max(abs(p - pattern)) <= r)
                C.append(count / (N-m+1))
            return np.mean(np.log(C))
        return abs(_phi(m+1) - _phi(m))
    
    extended['approximate_entropy'] = _approx_entropy(R)
    
    # Energy and envelope decay rates
    tau_env = results.get('tau_env_practical', results.get('tau_env_theorem', np.nan))
    tau_E = results.get('tau_E_practical', results.get('tau_E_theorem', np.nan))
    
    if tau_env is not None and not np.isnan(tau_env) and tau_env > 0:
        extended['envelope_decay_rate'] = 1 / tau_env
    else:
        extended['envelope_decay_rate'] = np.nan
    
    if tau_E is not None and not np.isnan(tau_E) and tau_E > 0:
        extended['energy_decay_rate'] = 1 / tau_E
    else:
        extended['energy_decay_rate'] = np.nan
    
    return extended


# ============================================================================
# MAIN QSPACE3D FUNCTION
# ============================================================================

def QSpace3d(
    t: np.ndarray,
    R: np.ndarray,
    R_inf: Optional[float] = None,
    normalize: bool = True,
    compute_method: Literal['practical', 'theorem', 'both'] = 'practical',
    Rinf_method: Literal['auto', 'integral', 'tail', 'last'] = 'auto',
    include_tau2: bool = False,
    include_extended: bool = False,
    oscillation_threshold: float = -0.2,
    delta_ps_threshold: float = 0.5,
    verbose: bool = False
) -> Dict[str, Any]:
    """Compute QSpace3d signature (X, Y, Z) for any time series."""
    
    # Input Validation
    t = np.asarray(t, dtype=float)
    R = np.asarray(R, dtype=float)
    R_raw = R.copy()
    
    if len(t) != len(R):
        return _empty_signature(f"t and R must have same length (got {len(t)} vs {len(R)})")
    if len(t) < 10:
        return _empty_signature(f"Need at least 10 points (got {len(t)})")
    if np.all(R == R[0]):
        return _empty_signature("response is constant — no dynamics to analyze")
    if np.any(np.isnan(t)) or np.any(np.isnan(R)):
        return _empty_signature("NaN values detected — please clean data first")
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        return _empty_signature("t must be strictly increasing")
    
    # Compute Δ_ps on RAW response
    delta_ps_raw = Delta_ps(t, R_raw, method='hilbert')
    
    # Special Case Detection
    is_special, special_case = _detect_special_case(t, R)
    if is_special:
        if verbose:
            print(f"Detected special case: {special_case} — using theoretical values")
        output = _apply_special_case(special_case, include_tau2)
        output['delta_ps'] = delta_ps_raw
        output['has_secondary_oscillation'] = not np.isnan(delta_ps_raw) and delta_ps_raw > delta_ps_threshold
        output['oscillation_strength'] = 'secondary' if output['has_secondary_oscillation'] else 'none'
        return output
    
    # Normalization
    R_processed = R.copy()
    normalization_applied = False
    
    if normalize:
        original_min = np.min(R_processed)
        original_max = np.max(R_processed)
        if original_max - original_min > 1e-10:
            R_processed = (R_processed - original_min) / (original_max - original_min)
            normalization_applied = True
            if verbose:
                print(f"Normalized: range [{original_min:.3f}, {original_max:.3f}] → [0, 1]")
    
    # Response Typing
    dt = np.mean(np.diff(t))
    dR_test = np.gradient(R_processed, dt)
    zero_crossings = np.sum((dR_test[:-1] * dR_test[1:]) < 0)
    is_oscillatory_response = zero_crossings > 5
    
    if is_oscillatory_response and abs(R_processed[-1] - R_processed[0]) < 0.1 * np.ptp(R_processed):
        response_type = 'oscillatory'
    elif abs(R_processed[-1] - R_processed[0]) > 0.5 * np.ptp(R_processed):
        response_type = 'step_like'
    else:
        response_type = 'arbitrary'
    
    # Estimate R_inf
    if R_inf is None:
        R_inf_used = estimate_Rinf(t, R_processed, method=Rinf_method)
    else:
        R_inf_used = R_inf
    
    # Compute Core Estimators
    results = compute_all(t, R_processed, R_inf=R_inf_used, method=compute_method, verbose=verbose)
    
    # Extract X
    tau_s_val = results.get('tau_s', np.nan)
    tau_u_val = results.get('tau_u', np.nan)
    if tau_u_val is not None and not np.isnan(tau_u_val) and tau_u_val != 0:
        X = (tau_s_val - tau_u_val) / tau_u_val
        X = np.clip(X, -3.5, 2.0)
    else:
        X = np.nan
    
    # Extract Y
    tau_3_val = results.get('tau_3', np.nan)
    if tau_3_val is not None and not np.isnan(tau_3_val) and tau_3_val != 0:
        Y = tau_s_val / tau_3_val
    else:
        Y = np.nan
    
    # Determine Primary Oscillatory
    is_oscillatory_primary = not np.isnan(X) and X < oscillation_threshold
    
    # Extract Z
    Z = np.nan
    if is_oscillatory_primary:
        if compute_method in ['practical', 'both']:
            tau_env_val = results.get('tau_env_practical', np.nan)
            tau_E_val = results.get('tau_E_practical', np.nan)
        else:
            tau_env_val = results.get('tau_env_theorem', np.nan)
            tau_E_val = results.get('tau_E_theorem', np.nan)
        
        if tau_env_val is not None and not np.isnan(tau_env_val) and tau_env_val != 0:
            if tau_E_val is not None and not np.isnan(tau_E_val):
                Z = (tau_env_val - tau_E_val) / tau_env_val
                Z = np.clip(Z, -1.0, 1.0)
    
    # Secondary Oscillation Detection
    has_secondary_oscillation = False
    oscillation_strength = 'none'
    if not np.isnan(delta_ps_raw):
        has_secondary_oscillation = delta_ps_raw > delta_ps_threshold
    
    if is_oscillatory_primary:
        oscillation_strength = 'primary'
    elif has_secondary_oscillation:
        oscillation_strength = 'secondary'
    
    # Optional τ₂
    tau_2_val = None
    if include_tau2:
        tau_2_val = results.get('tau_2', np.nan)
    
    # Confidence Metrics
    confidence = _compute_confidence_metrics(
        t, R_processed, tau_s_val, tau_u_val, tau_2_val, tau_3_val, Z, is_oscillatory_primary
    )
    
    # Validation
    issues, warnings_list, is_plausible = _validate_signature(X, Y, Z, tau_2_val, is_oscillatory_primary)
    
    # Assemble Output
    output = {
        'X': X,
        'Y': Y,
        'Z': Z,
        'signature_3d': np.array([X, Y, Z]) if not np.isnan(X) else np.array([np.nan, np.nan, np.nan]),
        'is_oscillatory': is_oscillatory_primary,
        'has_secondary_oscillation': has_secondary_oscillation,
        'oscillation_strength': oscillation_strength,
        'delta_ps': delta_ps_raw,
        'confidence': confidence,
        'validation': {
            'issues': issues,
            'warnings': warnings_list,
            'is_plausible': is_plausible,
            'needs_review': len(issues) > 0 or confidence['quality_level'] == 'low'
        },
        'method_used': {
            'tau_3_method': 'autocorr_centered',
            'tau_env_method': 'hilbert' if compute_method in ['practical', 'both'] else 'peak',
            'tau_E_method': 'autocorrelation' if compute_method in ['practical', 'both'] else 'peak',
            'Rinf_method': Rinf_method,
            'fallback_used': False,
            'special_case': None
        },
        'metadata': {
            'is_oscillatory': is_oscillatory_primary,
            'has_secondary_oscillation': has_secondary_oscillation,
            'oscillation_strength': oscillation_strength,
            'response_type': response_type,
            'normalization_applied': normalization_applied,
            'Rinf_used': float(R_inf_used) if R_inf_used is not None else np.nan,
            'Rinf_method': Rinf_method,
            'compute_method': compute_method,
            'n_points': len(t),
            'duration': t[-1] - t[0] if len(t) > 1 else 0,
            'oscillation_threshold': oscillation_threshold,
            'delta_ps_threshold': delta_ps_threshold,
            'conservative': False,
            'near_conservative': False
        }
    }
    
    if include_tau2 and tau_2_val is not None and not np.isnan(tau_2_val):
        output['tau_2'] = tau_2_val
    
    if include_extended:
        output['extended'] = _compute_extended_diagnostics(t, R_processed, results, tau_s_val, tau_u_val)
    
    if verbose:
        output['_raw'] = {
            'tau_s': tau_s_val,
            'tau_u': tau_u_val,
            'tau_3': tau_3_val,
            'tau_env': results.get('tau_env_practical', results.get('tau_env_theorem', np.nan)),
            'tau_E': results.get('tau_E_practical', results.get('tau_E_theorem', np.nan)),
            'delta_ps_raw': delta_ps_raw
        }
    
    return output


def _empty_signature(error_msg: str) -> Dict[str, Any]:
    """Return empty signature when computation fails."""
    return {
        'X': np.nan,
        'Y': np.nan,
        'Z': np.nan,
        'signature_3d': np.array([np.nan, np.nan, np.nan]),
        'is_oscillatory': False,
        'has_secondary_oscillation': False,
        'oscillation_strength': 'none',
        'delta_ps': np.nan,
        'confidence': {
            'window_ratio': 0,
            'window_adequate': False,
            'tau_consistency': np.nan,
            'estimators_agree': False,
            'tau_3_stable': False,
            'Y_reliable': False,
            'Z_meaningful': False,
            'quality_score': 0,
            'quality_level': 'invalid'
        },
        'validation': {
            'issues': [error_msg],
            'warnings': [],
            'is_plausible': False,
            'needs_review': True
        },
        'method_used': {
            'tau_3_method': 'none',
            'tau_env_method': 'none',
            'tau_E_method': 'none',
            'Rinf_method': 'none',
            'fallback_used': False,
            'special_case': 'error'
        },
        'metadata': {
            'is_oscillatory': False,
            'has_secondary_oscillation': False,
            'oscillation_strength': 'none',
            'response_type': 'invalid',
            'normalization_applied': False,
            'Rinf_used': np.nan,
            'Rinf_method': 'none',
            'compute_method': 'none',
            'n_points': 0,
            'duration': 0,
            'oscillation_threshold': -0.2,
            'delta_ps_threshold': 0.5,
            'conservative': False,
            'near_conservative': False,
            'error': error_msg
        }
    }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_signature(t: np.ndarray, R: np.ndarray, **kwargs) -> Tuple[float, float, float]:
    """Quick signature: returns only (X, Y, Z)."""
    result = QSpace3d(t, R, include_extended=False, **kwargs)
    return result['X'], result['Y'], result['Z']


def signature_with_confidence(t: np.ndarray, R: np.ndarray, **kwargs) -> Dict[str, Any]:
    """Signature with confidence but without extended diagnostics."""
    return QSpace3d(t, R, include_extended=False, **kwargs)


def full_analysis(t: np.ndarray, R: np.ndarray, **kwargs) -> Dict[str, Any]:
    """Complete analysis with all extended diagnostics."""
    return QSpace3d(t, R, include_extended=True, **kwargs)


# ============================================================================
# ENVELOPE GROWTH RATE Λ (Helper Function)
# ============================================================================

def compute_lambda(t, R, min_peaks=3):
    """
    Compute envelope growth rate Λ from peak detection.
    
    Parameters:
    -----------
    t : array
        Time points
    R : array
        Response signal (already smoothed)
    min_peaks : int, default=3
        Minimum number of peaks required for reliable Λ
    
    Returns:
    --------
    Lambda : float
        Envelope growth rate (nan if insufficient peaks)
    n_peaks : int
        Number of peaks detected
    reliable : bool
        True if n_peaks >= min_peaks
    """
    from scipy.signal import detrend
    
    # Detrend to isolate oscillatory component
    R_detrend = detrend(R)
    R_abs = np.abs(R_detrend)
    
    # Find peaks
    peaks = []
    for i in range(1, len(R_abs)-1):
        if R_abs[i] > R_abs[i-1] and R_abs[i] > R_abs[i+1]:
            peaks.append((t[i], R_abs[i]))
    
    n_peaks = len(peaks)
    
    if n_peaks < min_peaks:
        return np.nan, n_peaks, False
    
    t_peaks = np.array([p[0] for p in peaks])
    A_peaks = np.array([p[1] for p in peaks])
    
    # Compute logarithmic growth rates
    log_ratios = []
    for i in range(len(t_peaks)-1):
        dt = t_peaks[i+1] - t_peaks[i]
        if dt > 0:
            log_ratio = np.log(A_peaks[i+1] / (A_peaks[i] + 1e-10))
            log_ratios.append(log_ratio / dt)
    
    if len(log_ratios) == 0:
        return np.nan, n_peaks, False
    
    Lambda = np.mean(log_ratios)
    
    # Check stability (early vs late)
    mid = n_peaks // 2
    if mid >= 2:
        Lambda_early = np.mean(log_ratios[:mid])
        Lambda_late = np.mean(log_ratios[mid:])
        stable = abs(Lambda_early - Lambda_late) < 0.05
    else:
        stable = True
    
    return Lambda, n_peaks, (n_peaks >= min_peaks and stable)


def classify_regime(Delta_su, R_su):
    """
    Classify dynamical regime from Δ_su and R_su.
    
    Returns:
    --------
    regime : str
    """
    # Growth regime (R_su > 1)
    if R_su > 1:
        return 'GROWTH'
    
    # Exponential regime (R_su ≈ 1, Δ_su ≈ 0)
    if abs(Delta_su) < 1e-5 and abs(R_su - 1) < 1e-5:
        return 'EXPONENTIAL'
    
    # Fractional regime (slight positive Δ_su)
    if -1e-5 <= Delta_su <= 0.005 and 0.99999 <= R_su <= 1.005:
        return 'FRACTIONAL'
    
    # Underdamped regime
    if -0.98 <= Delta_su <= -0.63 and 0.02 <= R_su <= 0.37:
        return 'UNDERDAMPED'
    
    # Weakly damped regime
    if -2.55 <= Delta_su <= -0.97 and -1.55 <= R_su <= -0.02:
        return 'WEAKLY DAMPED'
    
    # Conservative oscillatory regime
    if -3.02 <= Delta_su <= -1.00 and -2.02 <= R_su <= 0:
        return 'CONSERVATIVE'
    
    return 'UNCERTAIN'


def classify_trend(R_su, Lambda, Lambda_reliable):
    """
    Classify amplitude trend from R_su and Λ.
    
    Returns:
    --------
    trend : str ('GROWING', 'DECAYING', 'STABLE', 'UNCERTAIN')
    confidence : str ('HIGH', 'MEDIUM', 'LOW')
    """
    if not Lambda_reliable:
        # Fall back to R_su only
        if R_su > 1:
            return 'GROWING', 'MEDIUM'
        elif R_su < 0:
            return 'DECAYING', 'MEDIUM'
        elif abs(R_su - 1) < 0.1:
            return 'STABLE', 'MEDIUM'
        else:
            return 'UNCERTAIN', 'LOW'
    
    # Λ reliable: use both
    if R_su > 1 and Lambda > 0:
        return 'GROWING', 'HIGH'
    elif R_su < 0 and Lambda < 0:
        return 'DECAYING', 'HIGH'
    elif abs(R_su - 1) < 0.1 and abs(Lambda) < 0.001:
        return 'STABLE', 'HIGH'
    elif R_su > 1 and Lambda < 0:
        return 'GROWING', 'LOW'  # Disagreement (phase issue)
    elif R_su < 0 and Lambda > 0:
        return 'DECAYING', 'LOW'   # Disagreement (phase issue)
    else:
        return 'UNCERTAIN', 'LOW'


# ============================================================================
# UPDATED Q_all FUNCTION
# ============================================================================

def Q_all(
    systems_dict,
    decimals: int = 4,
    include_metadata: bool = True,
    verbose: bool = False,
    normalize: bool = True,
    compute_method: Literal['practical', 'theorem', 'both'] = 'practical',
    Rinf_method: Literal['auto', 'integral', 'tail', 'last'] = 'auto',
    oscillation_threshold: float = -0.2,
    delta_ps_threshold: float = 0.5,
    R_inf_dict: Optional[Dict[str, float]] = None,
    include_lambda: bool = True  # NEW: default True
) -> pd.DataFrame:
    """
    Compute all estimators for multiple systems and return as DataFrame.
    
    NEW in v1.0:
        - Λ (envelope growth rate)
        - Regime classification
        - Amplitude trend with confidence
    
    Returns DataFrame with columns:
        τ_g, τ_s, τ_u, τ₂, τ₃, τ_pole, τ_env, τ_E,
        R_su, Δ_su, Λ, n_peaks, Λ_reliable,
        regime, amplitude_trend, confidence,
        Δₛᵤ (X), ρ₁₃ (Y), Δ₂₃ᵉⁿᵛ (Z), Δ_ps,
        Quality, Window_Ratio, response_Type (optional)
    """
    from .smooth import QSmooth
    
    all_data = {}
    
    for name, (t, R) in systems_dict.items():
        if verbose:
            print(f"Processing: {name}...")
        
        R_inf = None
        if R_inf_dict and name in R_inf_dict:
            R_inf = R_inf_dict[name]
        
        sig = QSpace3d(
            t, R,
            R_inf=R_inf,
            normalize=normalize,
            compute_method=compute_method,
            Rinf_method=Rinf_method,
            include_tau2=True,
            oscillation_threshold=oscillation_threshold,
            delta_ps_threshold=delta_ps_threshold,
            verbose=False
        )
        
        results = compute_all(t, R, R_inf=R_inf, method=compute_method, verbose=False)
        
        # Compute R_su from X
        X = sig.get('X', np.nan)
        R_su = X + 1 if not np.isnan(X) else np.nan
        Delta_su = X if not np.isnan(X) else np.nan
        
        # Compute Λ (envelope growth rate)
        Lambda = np.nan
        n_peaks = 0
        Lambda_reliable = False
        
        if include_lambda:
            # Apply QSmooth first
            qs = QSmooth()
            t_arr = np.asarray(t)
            R_arr = np.asarray(R)
            R_smooth = qs.savgol(t_arr, R_arr, window_frac=0.1, polyorder=3)
            Lambda, n_peaks, Lambda_reliable = compute_lambda(t_arr, R_smooth)
        
        # Classify regime and trend
        regime = classify_regime(Delta_su, R_su)
        trend, confidence = classify_trend(R_su, Lambda, Lambda_reliable)
        
        data = {
            # Core timescales
            'τ_g': results.get('tau_g', np.nan),
            'τ_s': results.get('tau_s', np.nan),
            'τ_u': results.get('tau_u', np.nan),
            'τ₂': results.get('tau_2', np.nan),
            'τ₃': results.get('tau_3', np.nan),
            'τ_pole': results.get('tau_pole', np.nan),
            'τ_env': results.get('tau_env_practical', results.get('tau_env_theorem', np.nan)),
            'τ_E': results.get('tau_E_practical', results.get('tau_E_theorem', np.nan)),
            
            # QSignature diagnostics
            'R_su': R_su,
            'Δ_su': Delta_su,
            'Λ': Lambda,
            'n_peaks': n_peaks,
            'Λ_reliable': Lambda_reliable,
            
            # Classification
            'regime': regime,
            'amplitude_trend': trend,
            'confidence': confidence,
            
            # QSpace3d outputs
            'Δₛᵤ (X)': X,
            'ρ₁₃ (Y)': sig.get('Y', np.nan),
            'Δ₂₃ᵉⁿᵛ (Z)': sig.get('Z', np.nan),
            'Δ_ps': sig.get('delta_ps', np.nan),
        }
        
        if include_metadata:
            data.update({
                'Quality': sig.get('confidence', {}).get('quality_level', 'unknown'),
                'Window_Ratio': sig.get('confidence', {}).get('window_ratio', np.nan),
                'response_Type': sig.get('metadata', {}).get('response_type', 'unknown'),
                'Oscillatory_Primary': sig.get('is_oscillatory', False),
                'Has_Secondary_Oscillation': sig.get('has_secondary_oscillation', False),
                'Oscillation_Strength': sig.get('oscillation_strength', 'none'),
            })
        
        all_data[name] = data
    
    df = pd.DataFrame(all_data)
    
    # Round numeric columns
    numeric_cols = ['τ_g', 'τ_s', 'τ_u', 'τ₂', 'τ₃', 'τ_pole', 'τ_env', 'τ_E',
                    'R_su', 'Δ_su', 'Λ', 'Δₛᵤ (X)', 'ρ₁₃ (Y)', 'Δ₂₃ᵉⁿᵛ (Z)', 
                    'Δ_ps', 'Window_Ratio', 'n_peaks']
    
    for col in numeric_cols:
        if col in df.index:
            df.loc[col] = pd.to_numeric(df.loc[col], errors='coerce').round(decimals)
    
    if verbose:
        print(f"\n✅ Processed {len(systems_dict)} systems")
        print(f"   Λ reliable for {sum(df.loc['Λ_reliable'])}/{len(systems_dict)} systems")
    
    return df


# ============================================================================
# UPDATED Q_one FUNCTION
# ============================================================================

def Q_one(
    t: np.ndarray,
    R: np.ndarray,
    name: str = "System",
    decimals: int = 4,
    normalize: bool = True,
    compute_method: Literal['practical', 'theorem', 'both'] = 'practical',
    Rinf_method: Literal['auto', 'integral', 'tail', 'last'] = 'auto',
    oscillation_threshold: float = -0.2,
    delta_ps_threshold: float = 0.5,
    R_inf: Optional[float] = None,
    verbose: bool = False,
    include_lambda: bool = True  # NEW: default True
) -> pd.Series:
    """
    Compute all estimators for a single system and return as Series.
    
    NEW in v1.0:
        - Λ (envelope growth rate)
        - Regime classification
        - Amplitude trend with confidence
    """
    df = Q_all(
        {name: (t, R)},
        decimals=decimals,
        include_metadata=True,
        verbose=verbose,
        normalize=normalize,
        compute_method=compute_method,
        Rinf_method=Rinf_method,
        oscillation_threshold=oscillation_threshold,
        delta_ps_threshold=delta_ps_threshold,
        R_inf_dict={name: R_inf} if R_inf is not None else None,
        include_lambda=include_lambda
    )
    return df[name] if name in df.columns else pd.Series()


def print_Q_all(df, title: str = "ALL ESTIMATORS FOR ALL SYSTEMS"):
    """Pretty print the Q_all DataFrame with new classification columns."""
    print(f"\n{'═'*120}")
    print(f"  {title}")
    print(f"{'═'*120}")
    
    # Select key columns for display
    display_cols = ['τ_s', 'τ_u', 'R_su', 'Δ_su', 'Λ', 'regime', 'amplitude_trend', 'confidence']
    available_cols = [c for c in display_cols if c in df.index]
    
    print(df.loc[available_cols].to_string())
    print(f"{'═'*120}\n")










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














"""
Synthetic Data Generators for QSignature theorems validation and experimentation.
Complete version with all methods properly defined.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gamma
from typing import Tuple, Dict, List, Optional, Union, Callable
import matplotlib.pyplot as plt

class QSynthetic:
    """
    Generate synthetic responses R(t) for 20+ dynamical regimes.
    Each method returns (t, R) and provides .info() for comprehensive guidance.
    """
    
    # ==================== INFO DECORATOR ====================
    
    def _info(equation: str, 
              param_descriptions: dict,
              examples: list = None,
              notes: str = ""):
        """Decorator to add comprehensive info to methods."""
        def decorator(func):
            def info():
                return {
                    'system_class': func.__name__.replace('_', ' ').title(),
                    'equation': equation,
                    'function': f"SyntheticSystems.{func.__name__}",
                    'parameters': param_descriptions,
                    'returns': "(t, R) - time array and response array",
                    'examples': examples or [],
                    'notes': notes,
                    'noise_adding': "Use noise_std parameter or add_measurement_noise()"
                }
            func.info = info
            return func
        return decorator
    
    # ==================== SYSTEM GENERATORS ====================
    
    @staticmethod
    @_info(
        equation="R(t) = R_∞ * [1 - exp(-(t/τ)ᵝ)]",
        param_descriptions={
            'tau': "Time constant (relaxation time). Default: 1.0",
            'R_inf': "Steady-state/final value. Default: 1.0",
            'noise_std': "Standard deviation of Gaussian noise. Default: 0.0",
            't_max': "Maximum time (if None, uses 10*tau). Default: None",
            'beta': "Stretching exponent (beta=1 → standard exponential). Default: 1.0"
        },
        examples=[
            "Basic: t, R = exponential_decay(tau=2.0)",
            "With noise: t, R = exponential_decay(tau=1.5, noise_std=0.05)",
            "Stretched: t, R = exponential_decay(tau=1.0, beta=0.8)"
        ],
        notes="Most common 1st-order relaxation system. For beta≠1, becomes stretched exponential."
    )
    def exponential_decay(tau: float = 1.0, 
                         R_inf: float = 1.0, 
                         noise_std: float = 0.0,
                         t_max: float = None,
                         beta: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """1st-order exponential system."""
        if t_max is None:
            t_max = 10 * tau
        t = np.linspace(0, t_max, 1000)
        
        if beta == 1.0:
            R = R_inf * (1 - np.exp(-t / tau))
        else:
            R = R_inf * (1 - np.exp(-(t / tau) ** beta))
        
        if noise_std > 0:
            R += noise_std * np.random.randn(len(t))
        return t, R
    
    @staticmethod
    @_info(
        equation="R(t) = R_∞ * [1 - e^{-αt} * cos(ω_d t + φ)]",
        param_descriptions={
            'alpha': "Damping coefficient (higher = faster damping). Default: 0.1",
            'omega_d': "Damped natural frequency (radians/time). Default: 5.0",
            'R_inf': "Steady-state value. Default: 1.0",
            'phi': "Phase offset (radians). Default: 0.0",
            'noise_std': "Standard deviation of Gaussian noise. Default: 0.0"
        },
        examples=[
            "Standard: t, R = underdamped_oscillator(alpha=0.2, omega_d=6.0)",
            "With phase: t, R = underdamped_oscillator(phi=np.pi/4)",
            "High damping: t, R = underdamped_oscillator(alpha=0.5, omega_d=2.0)"
        ],
        notes="Models underdamped harmonic oscillator. α² < ω₀² for oscillations."
    )
    def underdamped_oscillator(alpha: float = 0.1, 
                               omega_d: float = 5.0, 
                               R_inf: float = 1.0,
                               phi: float = 0.0,
                               noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Underdamped linear oscillator."""
        t = np.linspace(0, 10 / alpha, 2000)
        R = R_inf * (1 - np.exp(-alpha * t) * np.cos(omega_d * t + phi))
        
        if noise_std > 0:
            R += noise_std * np.random.randn(len(t))
        return t, R
    
    @staticmethod
    @_info(
        equation="R(t) = R_∞ * [1 - (λ₂e^{-λ₁t} - λ₁e^{-λ₂t})/(λ₂-λ₁)]",
        param_descriptions={
            'tau1': "First time constant (fast). Default: 1.0",
            'tau2': "Second time constant (slow). Default: 5.0",
            'R_inf': "Steady-state value. Default: 1.0",
            'critical': "If True, use critically damped solution. Default: False",
            'noise_std': "Standard deviation of Gaussian noise. Default: 0.0"
        },
        examples=[
            "Two time constants: t, R = overdamped_system(tau1=0.5, tau2=3.0)",
            "Critically damped: t, R = overdamped_system(tau1=2.0, critical=True)",
            "With noise: t, R = overdamped_system(noise_std=0.02)"
        ],
        notes="Models overdamped 2nd-order system or critically damped when critical=True."
    )
    def overdamped_system(tau1: float = 1.0, 
                         tau2: float = 5.0, 
                         R_inf: float = 1.0,
                         critical: bool = False,
                         noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Overdamped or critically damped system."""
        t_max = max(10 * tau1, 10 * tau2)
        t = np.linspace(0, t_max, 1500)
        
        if critical:
            omega0 = 1 / tau1
            R = R_inf * (1 - (1 + omega0 * t) * np.exp(-omega0 * t))
        else:
            lambda1 = 1 / tau1
            lambda2 = 1 / tau2
            R = R_inf * (1 - (lambda2 * np.exp(-lambda1 * t) - lambda1 * np.exp(-lambda2 * t)) 
                        / (lambda2 - lambda1))
        
        if noise_std > 0:
            R += noise_std * np.random.randn(len(t))
        return t, R
    
    @staticmethod
    @_info(
        equation="R(t) = R_∞ * [1 - Σ_n A_n cos(nω₀t)]",
        param_descriptions={
            'omega0': "Fundamental frequency (radians/time). Default: 2π",
            'R_inf': "Steady-state value. Default: 1.0",
            'modes': "Number of harmonic modes to include. Default: 1",
            'noise_std': "Standard deviation of Gaussian noise. Default: 0.0"
        },
        examples=[
            "Single mode: t, R = conservative_oscillator(omega0=3.0)",
            "Multiple modes: t, R = conservative_oscillator(modes=3)",
            "High frequency: t, R = conservative_oscillator(omega0=10.0)"
        ],
        notes="Undamped (conservative) system with harmonic oscillations."
    )
    def conservative_oscillator(omega0: float = 2 * np.pi,
                               R_inf: float = 1.0,
                               modes: int = 1,
                               noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Conservative (undamped) system."""
        t = np.linspace(0, 10 * 2 * np.pi / omega0, 2000)
        R = np.zeros_like(t)
        
        for n in range(1, modes + 1):
            amplitude = 1.0 / n if modes > 1 else 1.0
            R += amplitude * np.cos(n * omega0 * t)
        
        R = R_inf * (1 - R / modes)
        R += noise_std * np.random.randn(len(t))
        return t, R
    
    @staticmethod
    @_info(
        equation="R(t) = R_∞ * [1 - Σ_i A_i e^{-α_i t} cos(ω_i t + φ_i)]",
        param_descriptions={
            'n_modes': "Number of frequency modes. Default: 10",
            'base_freq': "Base frequency. Default: 1.0",
            'damping_spread': "Spread of damping coefficients. Default: 0.2",
            'R_inf': "Steady-state value. Default: 1.0",
            'noise_std': "Standard deviation of Gaussian noise. Default: 0.0"
        },
        examples=[
            "Standard: t, R = distributed_wave_system()",
            "Few modes: t, R = distributed_wave_system(n_modes=5)",
            "High frequency: t, R = distributed_wave_system(base_freq=2.0)"
        ],
        notes="Models distributed systems (e.g., wave equation solutions, PDEs)."
    )
    def distributed_wave_system(n_modes: int = 10,
                               base_freq: float = 1.0,
                               damping_spread: float = 0.2,
                               R_inf: float = 1.0,
                               noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Distributed system with multiple modes."""
        t = np.linspace(0, 50, 5000)
        R = np.zeros_like(t)
        
        freqs = base_freq * np.arange(1, n_modes + 1)
        alphas = damping_spread * 0.1 * (freqs / freqs[0])
        
        for i in range(n_modes):
            A = 1.0 / (i + 1)
            phi = np.random.uniform(0, 2 * np.pi)
            R += A * np.exp(-alphas[i] * t) * np.cos(2 * np.pi * freqs[i] * t + phi)
        
        R = R_inf * (1 - R / np.max(np.abs(R)))
        
        if noise_std > 0:
            R += noise_std * np.random.randn(len(t))
        return t, R
    
    @staticmethod
    @_info(
        equation="R(t) = R_∞ * [1 - t^{-β}]",
        param_descriptions={
            'beta': "Power-law exponent (β > 0). Default: 1.5",
            'R_inf': "Steady-state value. Default: 1.0",
            'cutoff': "Maximum time value (logarithmic spacing). Default: 100.0",
            'noise_std': "Standard deviation of Gaussian noise. Default: 0.0"
        },
        examples=[
            "Standard: t, R = powerlaw_decay(beta=1.5)",
            "Fast decay: t, R = powerlaw_decay(beta=2.5)",
            "Slow decay: t, R = powerlaw_decay(beta=0.8)"
        ],
        notes="Power-law/heavy-tailed response. Time is logarithmically spaced."
    )
    def powerlaw_decay(beta: float = 1.5,
                      R_inf: float = 1.0,
                      cutoff: float = 100.0,
                      noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Power-law/heavy-tailed system."""
        t = np.logspace(-2, np.log10(cutoff), 1000)
        R = R_inf * (1 - t ** (-beta))
        R[t < 0.1] = 0
        
        if noise_std > 0:
            R += noise_std * np.random.randn(len(t))
        return t, R
    
    @staticmethod
    @_info(
        equation="ẍ + 2αẋ + ω₀²x + εx³ = 0",
        param_descriptions={
            'alpha': "Linear damping coefficient. Default: 0.1",
            'omega0': "Linear natural frequency. Default: 5.0",
            'epsilon': "Nonlinearity strength (ε > 0 = hardening spring). Default: 0.1",
            'initial_amplitude': "Initial displacement. Default: 1.0",
            'R_inf': "Steady-state value. Default: 1.0",
            'noise_std': "Standard deviation of Gaussian noise. Default: 0.0"
        },
        examples=[
            "Weak nonlinearity: t, R = duffing_oscillator(epsilon=0.05)",
            "Strong nonlinearity: t, R = duffing_oscillator(epsilon=0.5)",
            "High frequency: t, R = duffing_oscillator(omega0=8.0)"
        ],
        notes="Nonlinear Duffing oscillator. Shows amplitude-frequency dependence."
    )
    def duffing_oscillator(alpha: float = 0.1,
                          omega0: float = 5.0,
                          epsilon: float = 0.1,
                          initial_amplitude: float = 1.0,
                          R_inf: float = 1.0,
                          noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Nonlinear Duffing oscillator."""
        def duffing_eq(t, y):
            x, v = y
            dxdt = v
            dvdt = -2 * alpha * v - omega0**2 * x - epsilon * x**3
            return [dxdt, dvdt]
        
        t_span = (0, 50)
        t_eval = np.linspace(0, 50, 5000)
        y0 = [initial_amplitude, 0]
        
        sol = solve_ivp(duffing_eq, t_span, y0, t_eval=t_eval, method='RK45')
        R = R_inf * (1 - sol.y[0] / initial_amplitude)
        
        if noise_std > 0:
            R += noise_std * np.random.randn(len(R))
        return sol.t, R
    
    @staticmethod
    @_info(
        equation="τ(t) = τ₀ + drift_rate * t (for linear drift)",
        param_descriptions={
            'tau0': "Initial time constant. Default: 1.0",
            'drift_rate': "Rate of parameter drift. Default: 0.01",
            'R_inf': "Steady-state value. Default: 1.0",
            'drift_type': "Type of drift: 'linear', 'exponential', or 'step'. Default: 'linear'",
            'noise_std': "Standard deviation of Gaussian noise. Default: 0.0"
        },
        examples=[
            "Linear aging: t, R = aging_system(drift_rate=0.02)",
            "Step change: t, R = aging_system(drift_type='step')",
            "Exponential aging: t, R = aging_system(drift_type='exponential')"
        ],
        notes="System with time-varying parameters (aging/degradation effects)."
    )
    def aging_system(tau0: float = 1.0,
                    drift_rate: float = 0.01,
                    R_inf: float = 1.0,
                    drift_type: str = 'linear',
                    noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Time-varying/aging system."""
        t = np.linspace(0, 100, 5000)
        
        if drift_type == 'linear':
            tau_t = tau0 + drift_rate * t
        elif drift_type == 'exponential':
            tau_t = tau0 * np.exp(drift_rate * t)
        else:
            tau_t = tau0 * np.ones_like(t)
            tau_t[t > 50] = tau0 * (1 + drift_rate)
        
        dt = t[1] - t[0]
        R = np.zeros_like(t)
        for i in range(1, len(t)):
            R[i] = R[i-1] + (R_inf - R[i-1]) * dt / tau_t[i]
        
        if noise_std > 0:
            R += noise_std * np.random.randn(len(t))
        return t, R
    
    @staticmethod
    @_info(
        equation="dR = (μ - R)/τ dt + σ dW (Ornstein-Uhlenbeck)",
        param_descriptions={
            'sigma': "Volatility/noise intensity. Default: 0.1",
            'tau': "Mean reversion time. Default: 2.0",
            'R_inf': "Long-term mean. Default: 1.0",
            'method': "SDE type: 'ornstein_uhlenbeck', 'geometric_brownian', or 'cir'. Default: 'ornstein_uhlenbeck'",
            't_max': "Maximum simulation time. Default: 20.0",
            'noise_std': "Additional measurement noise. Default: 0.0"
        },
        examples=[
            "OU process: t, R = stochastic_differential(method='ornstein_uhlenbeck')",
            "Geometric BM: t, R = stochastic_differential(method='geometric_brownian')",
            "CIR model: t, R = stochastic_differential(method='cir')"
        ],
        notes="Stochastic differential equations for modeling random processes."
    )
    def stochastic_differential(sigma: float = 0.1,
                               tau: float = 2.0,
                               R_inf: float = 1.0,
                               method: str = 'ornstein_uhlenbeck',
                               t_max: float = 20.0,
                               noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate responses from stochastic differential equations."""
        dt = 0.01
        n_steps = int(t_max / dt)
        t = np.linspace(0, t_max, n_steps)
        
        if method == 'ornstein_uhlenbeck':
            R = np.zeros(n_steps)
            R[0] = 0
            dW = np.sqrt(dt) * np.random.randn(n_steps-1)
            for i in range(1, n_steps):
                R[i] = R[i-1] + (R_inf - R[i-1])/tau * dt + sigma * dW[i-1]
        
        elif method == 'geometric_brownian':
            R = np.zeros(n_steps)
            R[0] = 1
            dW = np.sqrt(dt) * np.random.randn(n_steps-1)
            mu = np.log(R_inf) / t_max
            for i in range(1, n_steps):
                R[i] = R[i-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW[i-1])
        
        elif method == 'cir':
            R = np.zeros(n_steps)
            R[0] = 0.5
            dW = np.sqrt(dt) * np.random.randn(n_steps-1)
            a = 1/tau
            b = R_inf
            for i in range(1, n_steps):
                drift = a * (b - R[i-1]) * dt
                diffusion = sigma * np.sqrt(max(R[i-1], 0)) * dW[i-1]
                R[i] = max(R[i-1] + drift + diffusion, 0)
        
        if noise_std > 0:
            R += noise_std * np.random.randn(len(R))
        return t, R
    
    @staticmethod
    @_info(
        equation="ẋ = σ(y - x), ẏ = x(ρ - z) - y, ż = xy - βz (Lorenz)",
        param_descriptions={
            'system_type': "Chaotic system: 'lorenz' or 'rossler'. Default: 'lorenz'",
            'params': "Dictionary of system parameters. Default: None",
            'initial_conditions': "Initial state [x0, y0, z0]. Default: None",
            't_max': "Maximum simulation time. Default: 50.0",
            'noise_std': "Measurement noise. Default: 0.0"
        },
        examples=[
            "Lorenz: t, R = chaotic_system(system_type='lorenz')",
            "Rössler: t, R = chaotic_system(system_type='rossler')",
            "Custom: t, R = chaotic_system(params={'rho': 40, 'sigma': 16})"
        ],
        notes="Chaotic dynamical systems showing sensitive dependence on initial conditions."
    )
    def chaotic_system(system_type: str = 'lorenz',
                      params: Dict = None,
                      initial_conditions: List = None,
                      t_max: float = 50.0,
                      noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate responses from chaotic dynamical systems."""
        if params is None:
            params = {}
        
        def lorenz(t, y, sigma=10, rho=28, beta=8/3):
            return [sigma * (y[1] - y[0]),
                    y[0] * (rho - y[2]) - y[1],
                    y[0] * y[1] - beta * y[2]]
        
        def rossler(t, y, a=0.2, b=0.2, c=5.7):
            return [-y[1] - y[2],
                    y[0] + a * y[1],
                    b + y[2] * (y[0] - c)]
        
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, 10000)
        
        if system_type == 'lorenz':
            if initial_conditions is None:
                initial_conditions = [1.0, 1.0, 1.0]
            sol = solve_ivp(lorenz, t_span, initial_conditions, 
                           t_eval=t_eval, args=(params.get('sigma', 10),
                                              params.get('rho', 28),
                                              params.get('beta', 8/3)))
            R = sol.y[0]
        
        elif system_type == 'rossler':
            if initial_conditions is None:
                initial_conditions = [0.1, 0.1, 0.1]
            sol = solve_ivp(rossler, t_span, initial_conditions,
                           t_eval=t_eval, args=(params.get('a', 0.2),
                                              params.get('b', 0.2),
                                              params.get('c', 5.7)))
            R = sol.y[0]
        
        if noise_std > 0:
            R += noise_std * np.random.randn(len(R))
        return sol.t, R
    
    @staticmethod
    @_info(
        equation="D^α y(t) = f(t, y) where D^α is fractional derivative",
        param_descriptions={
            'alpha': "Fractional order (0 < α ≤ 2). Default: 0.8",
            'beta': "Additional Mittag-Leffler parameter. Default: 1.0",
            'method': "Solution method: 'mittag_leffler' or 'power_law'. Default: 'mittag_leffler'",
            't_max': "Maximum time. Default: 10.0",
            'noise_std': "Measurement noise. Default: 0.0"
        },
        examples=[
            "Standard: t, R = fractional_order_system(alpha=0.7)",
            "Different method: t, R = fractional_order_system(method='power_law')",
            "Higher order: t, R = fractional_order_system(alpha=1.3)"
        ],
        notes="Fractional order systems model memory effects and anomalous diffusion."
    )
    def fractional_order_system(alpha: float = 0.8,
                               beta: float = 1.0,
                               method: str = 'mittag_leffler',
                               t_max: float = 10.0,
                               noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate responses from fractional order differential equations."""
        t = np.linspace(0, t_max, 1000)
        
        def mittag_leffler(t, alpha, beta, n_terms=30):
            result = np.zeros_like(t)
            for k in range(n_terms):
                result += (t**k) / gamma(alpha*k + beta)
            return result
        
        if method == 'mittag_leffler':
            R = t**(alpha-1) * mittag_leffler(-t**alpha, alpha, alpha)
        elif method == 'power_law':
            R = t**(alpha-1) * np.exp(-t**alpha / gamma(alpha+1))
        
        R = R / np.max(R) if np.max(R) > 0 else R
        
        if noise_std > 0:
            R += noise_std * np.random.randn(len(R))
        return t, R
    
    @staticmethod
    @_info(
        equation="f(t) = f₀ + (f₁ - f₀) * t/T (linear chirp)",
        param_descriptions={
            'f0': "Initial frequency (Hz). Default: 1.0",
            'f1': "Final frequency (Hz). Default: 10.0",
            'chirp_type': "Type: 'linear', 'exponential', or 'logarithmic'. Default: 'linear'",
            'duration': "Chirp duration. Default: 10.0",
            'damping': "Exponential damping coefficient. Default: 0.0",
            'noise_std': "Measurement noise. Default: 0.0"
        },
        examples=[
            "Linear chirp: t, R = chirp_system(f0=1, f1=20)",
            "Exponential chirp: t, R = chirp_system(chirp_type='exponential')",
            "Damped chirp: t, R = chirp_system(damping=0.1)"
        ],
        notes="Signal with time-varying frequency (chirp). Useful for frequency response analysis."
    )
    def chirp_system(f0: float = 1.0,
                    f1: float = 10.0,
                    chirp_type: str = 'linear',
                    duration: float = 10.0,
                    damping: float = 0.0,
                    noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate responses with time-varying frequencies."""
        t = np.linspace(0, duration, int(1000 * duration))
        
        if chirp_type == 'linear':
            phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t**2 / duration)
        elif chirp_type == 'exponential':
            k = np.log(f1/f0) / duration
            phase = 2 * np.pi * f0 * (np.exp(k * t) - 1) / k
        else:
            phase = 2 * np.pi * f0 * duration * np.log(1 + (f1/f0 - 1) * t/duration) / np.log(f1/f0)
        
        R = np.cos(phase)
        
        if damping > 0:
            R = R * np.exp(-damping * t)
        
        if noise_std > 0:
            R += noise_std * np.random.randn(len(R))
        return t, R
    
    @staticmethod
    @_info(
        equation="R(t) = Σ_i w_i * e^{-t/τ_i} * cos(ω_i t)",
        param_descriptions={
            'fast_tau': "Fastest time scale. Default: 0.1",
            'slow_tau': "Slowest time scale. Default: 5.0",
            'coupling_strength': "Interaction between scales. Default: 0.3",
            'n_scales': "Number of temporal scales. Default: 3",
            'noise_std': "Measurement noise. Default: 0.0"
        },
        examples=[
            "Standard: t, R = multiscale_system()",
            "More scales: t, R = multiscale_system(n_scales=5)",
            "Strong coupling: t, R = multiscale_system(coupling_strength=0.8)"
        ],
        notes="System with multiple interacting temporal scales (hierarchical dynamics)."
    )
    def multiscale_system(fast_tau: float = 0.1,
                         slow_tau: float = 5.0,
                         coupling_strength: float = 0.3,
                         n_scales: int = 3,
                         noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate responses with multiple temporal scales."""
        t_max = 10 * slow_tau
        t = np.linspace(0, t_max, 5000)
        
        taus = np.logspace(np.log10(fast_tau), np.log10(slow_tau), n_scales)
        
        responses = []
        for i, tau in enumerate(taus):
            freq = 2 * np.pi / tau
            R_i = np.exp(-t / tau) * np.cos(freq * t)
            
            if i > 0 and coupling_strength > 0:
                R_i = R_i * (1 + coupling_strength * responses[-1])
            
            responses.append(R_i)
        
        weights = 1 / taus
        weights = weights / np.sum(weights)
        
        R = np.zeros_like(t)
        for i, R_i in enumerate(responses):
            R += weights[i] * R_i
        
        R = (R - np.min(R)) / (np.max(R) - np.min(R) + 1e-10)
        
        if noise_std > 0:
            R += noise_std * np.random.randn(len(R))
        return t, R
    
    @staticmethod
    @_info(
        equation="R(t) = background(t) + Σ bursts(t - t_i)",
        param_descriptions={
            'burst_rate': "Average bursts per unit time. Default: 0.1",
            'burst_duration': "Average duration of bursts. Default: 1.0",
            'background_noise': "Background process intensity. Default: 0.05",
            'burst_amplitude': "Typical burst amplitude. Default: 2.0",
            't_max': "Maximum simulation time. Default: 100.0",
            'noise_std': "Additional measurement noise. Default: 0.0"
        },
        examples=[
            "Standard: t, R = intermittent_system()",
            "Frequent bursts: t, R = intermittent_system(burst_rate=0.3)",
            "Large bursts: t, R = intermittent_system(burst_amplitude=5.0)"
        ],
        notes="Intermittent/bursty system with quiescent periods and sudden bursts."
    )
    def intermittent_system(burst_rate: float = 0.1,
                           burst_duration: float = 1.0,
                           background_noise: float = 0.05,
                           burst_amplitude: float = 2.0,
                           t_max: float = 100.0,
                           noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate responses with intermittent bursts."""
        dt = 0.01
        t = np.linspace(0, t_max, int(t_max/dt))
        
        # Create background process (Ornstein-Uhlenbeck)
        R = np.zeros_like(t)
        tau = 5.0
        R[0] = 0
        for i in range(1, len(t)):
            R[i] = R[i-1] - R[i-1]/tau * dt + background_noise * np.sqrt(dt) * np.random.randn()
        
        # Add bursts
        n_bursts = int(burst_rate * t_max)
        burst_times = np.random.uniform(0, t_max, n_bursts)
        
        for burst_time in burst_times:
            idx = int(burst_time / dt)
            burst_length = int(burst_duration / dt)
            
            if idx + burst_length < len(t):
                t_burst = np.linspace(0, burst_duration, burst_length)
                burst = burst_amplitude * np.exp(-t_burst/0.2) * np.sin(2*np.pi*5*t_burst)
                R[idx:idx+burst_length] += burst
        
        if noise_std > 0:
            R += noise_std * np.random.randn(len(R))
        return t, R
    
    # ==================== UTILITY FUNCTIONS ====================
    
    @staticmethod
    @_info(
        equation="R_noisy = R + noise",
        param_descriptions={
            'R': "Original clean response array (required)",
            'noise_type': "Type of noise: 'gaussian', 'poisson', 'uniform', 'salt_and_pepper'. Default: 'gaussian'",
            '**kwargs': "Noise parameters (std for gaussian, amplitude for uniform, etc.)"
        },
        examples=[
            "Gaussian: R_noisy = add_measurement_noise(R, noise_type='gaussian', std=0.05)",
            "Poisson: R_noisy = add_measurement_noise(R, noise_type='poisson', scaling=100)",
            "Salt & Pepper: R_noisy = add_measurement_noise(R, noise_type='salt_and_pepper', probability=0.01)"
        ],
        notes="Add realistic measurement noise to synthetic data."
    )
    def add_measurement_noise(R: np.ndarray,
                             noise_type: str = 'gaussian',
                             **kwargs) -> np.ndarray:
        """Add various types of measurement noise."""
        R_noisy = R.copy()
        
        if noise_type == 'gaussian':
            std = kwargs.get('std', 0.1)
            R_noisy += np.random.randn(len(R)) * std
        
        elif noise_type == 'poisson':
            scaling = kwargs.get('scaling', 100)
            R_scaled = np.maximum(R, 0) * scaling
            R_noisy = np.random.poisson(np.maximum(R_scaled, 0)) / scaling
        
        elif noise_type == 'uniform':
            amplitude = kwargs.get('amplitude', 0.2)
            R_noisy += np.random.uniform(-amplitude, amplitude, len(R))
        
        elif noise_type == 'salt_and_pepper':
            probability = kwargs.get('probability', 0.01)
            mask = np.random.random(len(R)) < probability/2
            R_noisy[mask] = kwargs.get('salt_value', np.max(R))
            mask = np.random.random(len(R)) < probability/2
            R_noisy[mask] = kwargs.get('pepper_value', np.min(R))
        
        return R_noisy
    
    @staticmethod
    @_info(
        equation="t_irregular = t + jitter, then remove random points",
        param_descriptions={
            't': "Original time array (required)",
            'R': "Original response array (required)",
            'missing_prob': "Probability of missing each data point. Default: 0.05",
            'jitter_std': "Standard deviation of time jitter. Default: 0.01"
        },
        examples=[
            "Standard: t_irr, R_irr = add_sampling_irregularities(t, R)",
            "More missing: t_irr, R_irr = add_sampling_irregularities(t, R, missing_prob=0.1)",
            "With jitter: t_irr, R_irr = add_sampling_irregularities(t, R, jitter_std=0.05)"
        ],
        notes="Simulate realistic sampling issues: missing data and time measurement errors."
    )
    def add_sampling_irregularities(t: np.ndarray,
                                   R: np.ndarray,
                                   missing_prob: float = 0.05,
                                   jitter_std: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Introduce realistic sampling issues."""
        t_jittered = t + np.random.randn(len(t)) * jitter_std
        t_jittered = np.sort(t_jittered)
        
        keep_mask = np.random.random(len(t)) > missing_prob
        t_irregular = t_jittered[keep_mask]
        R_irregular = R[keep_mask]
        
        return t_irregular, R_irregular
    
    @staticmethod
    def get_all_system_types() -> Dict:
        """Return dictionary of all available system generators."""
        return {
            'exponential': {
                'generator': SyntheticSystems.exponential_decay,
                'info': SyntheticSystems.exponential_decay.info
            },
            'underdamped': {
                'generator': SyntheticSystems.underdamped_oscillator,
                'info': SyntheticSystems.underdamped_oscillator.info
            },
            'overdamped': {
                'generator': SyntheticSystems.overdamped_system,
                'info': SyntheticSystems.overdamped_system.info
            },
            'conservative': {
                'generator': SyntheticSystems.conservative_oscillator,
                'info': SyntheticSystems.conservative_oscillator.info
            },
            'distributed': {
                'generator': SyntheticSystems.distributed_wave_system,
                'info': SyntheticSystems.distributed_wave_system.info
            },
            'powerlaw': {
                'generator': SyntheticSystems.powerlaw_decay,
                'info': SyntheticSystems.powerlaw_decay.info
            },
            'duffing': {
                'generator': SyntheticSystems.duffing_oscillator,
                'info': SyntheticSystems.duffing_oscillator.info
            },
            'aging': {
                'generator': SyntheticSystems.aging_system,
                'info': SyntheticSystems.aging_system.info
            },
            'stochastic': {
                'generator': SyntheticSystems.stochastic_differential,
                'info': SyntheticSystems.stochastic_differential.info
            },
            'chaotic': {
                'generator': SyntheticSystems.chaotic_system,
                'info': SyntheticSystems.chaotic_system.info
            },
            'fractional': {
                'generator': SyntheticSystems.fractional_order_system,
                'info': SyntheticSystems.fractional_order_system.info
            },
            'chirp': {
                'generator': SyntheticSystems.chirp_system,
                'info': SyntheticSystems.chirp_system.info
            },
            'multiscale': {
                'generator': SyntheticSystems.multiscale_system,
                'info': SyntheticSystems.multiscale_system.info
            },
            'intermittent': {
                'generator': SyntheticSystems.intermittent_system,
                'info': SyntheticSystems.intermittent_system.info
            }
        }
    
    @staticmethod
    def generate_dataset(n_samples: int = 100, 
                        add_noise: bool = True,
                        noise_std: float = 0.05) -> Dict:
        """Generate a diverse dataset of synthetic responses."""
        
        systems = SyntheticSystems.get_all_system_types()
        system_keys = list(systems.keys())
        
        dataset = {
            't': [],
            'R': [],
            'R_clean': [],
            'system_type': [],
            'parameters': []
        }
        
        for i in range(n_samples):
            sys_type = np.random.choice(system_keys)
            generator = systems[sys_type]['generator']
            
            params = {}
            
            if sys_type == 'exponential':
                params = {
                    'tau': np.random.uniform(0.5, 5.0),
                    'R_inf': np.random.uniform(0.5, 2.0),
                    'noise_std': 0.0
                }
            
            elif sys_type == 'underdamped':
                params = {
                    'alpha': np.random.uniform(0.05, 0.5),
                    'omega_d': np.random.uniform(2.0, 15.0),
                    'R_inf': np.random.uniform(0.5, 2.0),
                    'noise_std': 0.0
                }
            
            elif sys_type == 'overdamped':
                params = {
                    'tau1': np.random.uniform(0.5, 2.0),
                    'tau2': np.random.uniform(2.0, 10.0),
                    'R_inf': np.random.uniform(0.5, 2.0),
                    'noise_std': 0.0
                }
            
            elif sys_type == 'stochastic':
                params = {
                    'sigma': np.random.uniform(0.05, 0.3),
                    'tau': np.random.uniform(1.0, 5.0),
                    'R_inf': np.random.uniform(0.5, 2.0),
                    'method': np.random.choice(['ornstein_uhlenbeck', 'geometric_brownian']),
                    't_max': np.random.uniform(10, 30),
                    'noise_std': 0.0
                }
            
            elif sys_type == 'chaotic':
                params = {
                    'system_type': np.random.choice(['lorenz', 'rossler']),
                    't_max': np.random.uniform(20, 50),
                    'noise_std': 0.0
                }
            
            elif sys_type == 'fractional':
                params = {
                    'alpha': np.random.uniform(0.5, 1.5),
                    't_max': np.random.uniform(5, 15),
                    'noise_std': 0.0
                }
            
            elif sys_type == 'chirp':
                params = {
                    'f0': np.random.uniform(0.5, 3.0),
                    'f1': np.random.uniform(5.0, 20.0),
                    'chirp_type': np.random.choice(['linear', 'exponential']),
                    'duration': np.random.uniform(5, 20),
                    'damping': np.random.uniform(0, 0.2),
                    'noise_std': 0.0
                }
            
            elif sys_type == 'multiscale':
                params = {
                    'fast_tau': np.random.uniform(0.05, 0.5),
                    'slow_tau': np.random.uniform(3.0, 15.0),
                    'coupling_strength': np.random.uniform(0.1, 0.5),
                    'n_scales': np.random.choice([3, 4, 5]),
                    'noise_std': 0.0
                }
            
            elif sys_type == 'intermittent':
                params = {
                    'burst_rate': np.random.uniform(0.05, 0.3),
                    'burst_duration': np.random.uniform(0.5, 3.0),
                    'background_noise': np.random.uniform(0.02, 0.1),
                    'burst_amplitude': np.random.uniform(1.0, 3.0),
                    't_max': np.random.uniform(50, 150),
                    'noise_std': 0.0
                }
            
            elif sys_type == 'aging':
                params = {
                    'tau0': np.random.uniform(0.5, 5.0),
                    'drift_rate': np.random.uniform(0.005, 0.05),
                    'R_inf': np.random.uniform(0.5, 2.0),
                    'drift_type': np.random.choice(['linear', 'exponential', 'step']),
                    'noise_std': 0.0
                }
            
            elif sys_type == 'powerlaw':
                params = {
                    'beta': np.random.uniform(0.8, 2.5),
                    'R_inf': np.random.uniform(0.5, 2.0),
                    'cutoff': np.random.uniform(50, 200),
                    'noise_std': 0.0
                }
            
            else:
                params = {'noise_std': 0.0}
            
            t, R_clean = generator(**params)
            
            if add_noise:
                R = R_clean + noise_std * np.random.randn(len(R_clean))
            else:
                R = R_clean.copy()
            
            dataset['t'].append(t)
            dataset['R'].append(R)
            dataset['R_clean'].append(R_clean)
            dataset['system_type'].append(sys_type)
            dataset['parameters'].append(params)
        
        return dataset
    
    # ==================== HELPER FUNCTIONS ====================
    
    @staticmethod
    def pretty_info(func_name: str) -> None:
        """Print comprehensive info about a function in a nice format."""
        if not hasattr(SyntheticSystems, func_name):
            print(f"Function '{func_name}' not found!")
            return
        
        func = getattr(SyntheticSystems, func_name)
        
        if not hasattr(func, 'info'):
            print(f"Function '{func_name}' doesn't have info.")
            return
        
        info = func.info()
        
        print("\n" + "="*80)
        print(f"📘 {func_name.replace('_', ' ').title()}")
        print("="*80)
        
        print(f"\n📝 Equation:")
        print(f"   {info['equation']}")
        
        print(f"\n🔧 Function Call:")
        print(f"   {info['function']}")
        
        print(f"\n⚙️ Parameters:")
        for param, desc in info['parameters'].items():
            print(f"   • {param:20} - {desc}")
        
        print(f"\n📤 Returns:")
        print(f"   {info['returns']}")
        
        if info['examples']:
            print(f"\n🎯 Examples:")
            for ex in info['examples']:
                print(f"   {ex}")
        
        if info['notes']:
            print(f"\n💡 Notes:")
            print(f"   {info['notes']}")
        
        print(f"\n🎚️ Adding Noise:")
        print(f"   {info['noise_adding']}")
        
        print("\n" + "="*80)






# ============================================================================
# EXPORT ALL FUNCTIONS
# ============================================================================

__all__ = [
    'tau_g', 'tau_s', 'tau_u', 'tau_2', 'tau_3', 'tau_pole',
    'tau_env', 'tau_E', 'Delta_su', 'Delta_23_env', 'Delta_ps',
    'rho_13_step', 'rho_13_impulse',
    'compute_all', 'print_summary', 'validate_theorems',
    'QSpace3d', 'quick_signature', 'signature_with_confidence', 'full_analysis',
    'Q_all', 'print_Q_all', 'Q_one',
    'normalize_response', 'detect_response_type', 'validate_input', 'responseType',
    'estimate_Rinf',
    # Add these:
    'QSmooth',
    'QSynthetic',
]