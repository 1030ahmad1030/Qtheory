"""
Synthetic Data Generators for QSignature theorems validation and experimentation.
Complete version with all methods properly defined.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gamma
from typing import Tuple, Dict, List, Optional, Union, Callable
import matplotlib.pyplot as plt

class SyntheticSystems:
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

# ==================== TEST ====================

if __name__ == "__main__":
    print("Testing SyntheticSystems class...\n")
    
    # Test duffing oscillator
    t, R = SyntheticSystems.duffing_oscillator(
        omega0=8.0, alpha=0.5, epsilon=0.5, initial_amplitude=2
    )
    
    print(f"✓ duffing_oscillator works!")
    print(f"  Generated {len(t)} points")
    print(f"  t range: [{t[0]:.1f}, {t[-1]:.1f}]")
    print(f"  R range: [{R.min():.3f}, {R.max():.3f}]")
    
    # Test info
    print(f"\n✓ duffing_oscillator.info() works:")
    info = SyntheticSystems.duffing_oscillator.info()
    print(f"  System: {info['system_class']}")
    print(f"  Equation: {info['equation']}")
    print(f"  Parameters: {len(info['parameters'])}")