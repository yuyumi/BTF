import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List, Optional, Callable, Protocol
from abc import ABC, abstractmethod
import math
import json
import os


class WeightSelector(Protocol):
    """Protocol for weight selection strategies."""
    
    def compute_weights(self, n: int, K: float, tau: float, J_n: int) -> Dict[int, float]:
        """
        Compute mixture weights w_{j,n} for all resolution levels.
        
        Args:
            n: Sample size
            K: Lower bound parameter (n^{-K})
            tau: Sparsity parameter (2^{-j(1+τ)})
            J_n: Resolution cutoff
            
        Returns:
            Dictionary mapping j -> w_{j,n}
        """
        ...


class SlabDistribution(Protocol):
    """Protocol for slab distribution g(x)."""
    
    def sample(self) -> float:
        """Sample from g(x)."""
        ...
    
    def log_prob(self, x: float) -> float:
        """Compute log g(x)."""
        ...
    
    def verify_boundedness(self, L0: float) -> bool:
        """Verify boundedness condition: inf_{x∈[-L₀,L₀]} g(x) > 0."""
        ...


# Weight Selection Strategies
class GeometricMeanWeights:
    """Geometric mean between bounds: w_{j,n} = sqrt(n^{-K} * 2^{-j(1+τ)})."""
    
    def compute_weights(self, n: int, K: float, tau: float, J_n: int) -> Dict[int, float]:
        weights = {}
        lower_bound = n ** (-K)
        
        for j in range(J_n + 1):
            upper_bound = 2 ** (-j * (1 + tau))
            if upper_bound >= lower_bound:
                weights[j] = math.sqrt(lower_bound * upper_bound)
            else:
                weights[j] = lower_bound
                
        return weights


class UniformWeights:
    """Uniformly sample weights within bounds."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
    
    def compute_weights(self, n: int, K: float, tau: float, J_n: int) -> Dict[int, float]:
        weights = {}
        lower_bound = n ** (-K)
        
        for j in range(J_n + 1):
            upper_bound = 2 ** (-j * (1 + tau))
            if upper_bound >= lower_bound:
                weights[j] = self.rng.uniform(lower_bound, upper_bound)
            else:
                weights[j] = lower_bound
                
        return weights


class ArithmeticMeanWeights:
    """Arithmetic mean between bounds: w_{j,n} = (n^{-K} + 2^{-j(1+τ)}) / 2."""
    
    def compute_weights(self, n: int, K: float, tau: float, J_n: int) -> Dict[int, float]:
        weights = {}
        lower_bound = n ** (-K)
        
        for j in range(J_n + 1):
            upper_bound = 2 ** (-j * (1 + tau))
            if upper_bound >= lower_bound:
                weights[j] = (lower_bound + upper_bound) / 2
            else:
                weights[j] = lower_bound
                
        return weights


class ExponentialDecayWeights:
    """Exponential decay: w_{j,n} = w_0 * exp(-α*j) clamped to bounds."""
    
    def __init__(self, w0: float = 0.5, alpha: float = 0.5):
        self.w0 = w0
        self.alpha = alpha
    
    def compute_weights(self, n: int, K: float, tau: float, J_n: int) -> Dict[int, float]:
        weights = {}
        lower_bound = n ** (-K)
        
        for j in range(J_n + 1):
            upper_bound = 2 ** (-j * (1 + tau))
            unclamped_weight = self.w0 * math.exp(-self.alpha * j)
            
            # Clamp to bounds
            weights[j] = max(lower_bound, min(upper_bound, unclamped_weight))
                
        return weights


# Slab Distributions
class NormalSlab:
    """Normal slab distribution: g(x) ~ N(0, σ²)."""
    
    def __init__(self, std: float = 1.0):
        self.std = std
        self.dist = torch.distributions.Normal(0.0, std)
    
    def sample(self) -> float:
        return self.dist.sample().item()
    
    def log_prob(self, x: float) -> float:
        return self.dist.log_prob(torch.tensor(x)).item()
    
    def verify_boundedness(self, L0: float) -> bool:
        # Normal distribution is always positive
        x_test = torch.linspace(-L0, L0, 100)
        densities = torch.exp(self.dist.log_prob(x_test))
        return torch.min(densities).item() > 0


class LaplaceSlab:
    """Laplace slab distribution: g(x) ~ Laplace(0, b)."""
    
    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self.dist = torch.distributions.Laplace(0.0, scale)
    
    def sample(self) -> float:
        return self.dist.sample().item()
    
    def log_prob(self, x: float) -> float:
        return self.dist.log_prob(torch.tensor(x)).item()
    
    def verify_boundedness(self, L0: float) -> bool:
        # Laplace distribution is always positive
        return True


class UniformSlab:
    """Uniform slab distribution: g(x) ~ Uniform(-a, a)."""
    
    def __init__(self, half_width: float = 2.0):
        self.half_width = half_width
        self.dist = torch.distributions.Uniform(-half_width, half_width)
    
    def sample(self) -> float:
        return self.dist.sample().item()
    
    def log_prob(self, x: float) -> float:
        return self.dist.log_prob(torch.tensor(x)).item()
    
    def verify_boundedness(self, L0: float) -> bool:
        # Uniform distribution is constant on its support
        return self.half_width >= L0


class StudentTSlab:
    """Student-t slab distribution: g(x) ~ t_ν(0, σ²)."""
    
    def __init__(self, df: float = 3.0, scale: float = 1.0):
        self.df = df
        self.scale = scale
        self.dist = torch.distributions.StudentT(df, 0.0, scale)
    
    def sample(self) -> float:
        return self.dist.sample().item()
    
    def log_prob(self, x: float) -> float:
        return self.dist.log_prob(torch.tensor(x)).item()
    
    def verify_boundedness(self, L0: float) -> bool:
        # Student-t distribution is always positive (for finite df)
        if self.df <= 0:
            return False
        x_test = torch.linspace(-L0, L0, 100)
        densities = torch.exp(self.dist.log_prob(x_test))
        return torch.min(densities).item() > 0


class SpikeSlabPrior:
    """
    Modular spike and slab prior for wavelet coefficients.
    
    Implements the prior:
    π_j(dx) = (1 - w_{j,n})δ₀(dx) + w_{j,n}g(x)dx
    
    with constraints:
    - n^{-K} ≤ w_{j,n} ≤ 2^{-j(1+τ)} for j ≤ J_n
    - w_{j,n} = 0 for j > J_n
    - J_n = [log n / log 2]
    """
    
    def __init__(self, 
                 n: int,
                 K: Optional[float] = None,
                 tau: Optional[float] = None,
                 L0: Optional[float] = None,
                 max_resolution: Optional[int] = None,
                 weight_selector: Optional[WeightSelector] = None,
                 slab_distribution: Optional[SlabDistribution] = None):
        """
        Initialize modular spike-slab prior.
        
        Args:
            n: Sample size (determines resolution cutoff)
            K: Lower bound parameter for weights (n^{-K})
            tau: Sparsity parameter (controls 2^{-j(1+τ)})
            L0: Boundedness parameter for g(x)
            max_resolution: Maximum resolution level (if None, uses J_n)
            weight_selector: Strategy for computing mixture weights
            slab_distribution: Distribution for slab component g(x)
        """
        # Load global config - fail loudly if not found
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {e}")
        
        spike_slab_config = config.get('spike_slab_prior')
        if spike_slab_config is None:
            raise KeyError("'spike_slab_prior' section not found in config.json")
        
        # Use config values or provided parameters
        try:
            self.K = K if K is not None else spike_slab_config['K']
            self.tau = tau if tau is not None else spike_slab_config['tau']
            self.L0 = L0 if L0 is not None else spike_slab_config['L0']
        except KeyError as e:
            raise KeyError(f"Missing required key in spike_slab_prior config: {e}")
        
        # Ensure no None values - fail loudly if config is incomplete
        if self.L0 is None:
            raise ValueError("L0 parameter is None after loading config - check config.json")
        if self.K is None:
            raise ValueError("K parameter is None after loading config - check config.json")
        if self.tau is None:
            raise ValueError("tau parameter is None after loading config - check config.json")
        
        self.n = n
        
        # Resolution cutoff: J_n = [log n / log 2]
        self.J_n = int(math.log(n) / math.log(2)) if max_resolution is None else max_resolution
        
        # Default strategies if not provided
        self.weight_selector = weight_selector or GeometricMeanWeights()
        self.slab_distribution = slab_distribution or NormalSlab(std=1.0)
        
        # Precompute weights w_{j,n} for all resolution levels
        self.weights = self.weight_selector.compute_weights(n, self.K, self.tau, self.J_n)
        
        # Verify boundedness condition
        if not self.slab_distribution.verify_boundedness(self.L0):
            raise ValueError(f"Slab distribution does not satisfy boundedness condition for L0={self.L0}")
    
    def get_weight(self, j: int) -> float:
        """Get mixture weight w_{j,n} for resolution level j."""
        if j > self.J_n:
            return 0.0
        return self.weights.get(j, 0.0)
    
    def sample_coefficient(self, j: int, k: int) -> float:
        """
        Sample a single wavelet coefficient θ_{j,k} from π_j.
        
        Args:
            j: Resolution level
            k: Spatial index
            
        Returns:
            Sampled coefficient
        """
        w_jn = self.get_weight(j)
        
        if w_jn == 0.0:
            # Pure spike: θ_{j,k} = 0
            return 0.0
        
        # Bernoulli trial: spike vs slab
        if torch.rand(1).item() < (1 - w_jn):
            # Spike: θ_{j,k} = 0
            return 0.0
        else:
            # Slab: θ_{j,k} ~ g(x)
            return self.slab_distribution.sample()
    
    def sample_coefficients(self, resolution_structure: Dict[int, int]) -> Dict[Tuple[int, int], float]:
        """
        Sample all wavelet coefficients for given resolution structure.
        
        Args:
            resolution_structure: Dict mapping j -> |I_j| (number of coefficients at level j)
            
        Returns:
            Dict mapping (j,k) -> θ_{j,k}
        """
        coefficients = {}
        
        for j, num_coeffs in resolution_structure.items():
            for k in range(num_coeffs):
                coefficients[(j, k)] = self.sample_coefficient(j, k)
                
        return coefficients
    
    def log_prob_coefficient(self, j: int, k: int, theta_jk: float) -> float:
        """
        Compute log probability of coefficient θ_{j,k} under π_j.
        
        Args:
            j: Resolution level
            k: Spatial index  
            theta_jk: Coefficient value
            
        Returns:
            Log probability
        """
        w_jn = self.get_weight(j)
        
        if w_jn == 0.0:
            # Pure spike
            return 0.0 if theta_jk == 0.0 else -float('inf')
        
        if theta_jk == 0.0:
            # Could be from spike (slab has zero density at 0 for continuous distributions)
            return math.log(1 - w_jn)
        else:
            # Must be from slab
            slab_log_prob = self.slab_distribution.log_prob(theta_jk)
            return math.log(w_jn) + slab_log_prob
    
    def log_prob_coefficients(self, coefficients: Dict[Tuple[int, int], float]) -> float:
        """
        Compute total log probability of all coefficients.
        
        Args:
            coefficients: Dict mapping (j,k) -> θ_{j,k}
            
        Returns:
            Total log probability
        """
        total_log_prob = 0.0
        
        for (j, k), theta_jk in coefficients.items():
            total_log_prob += self.log_prob_coefficient(j, k, theta_jk)
            
        return total_log_prob
    
    def get_sparsity_info(self) -> Dict[str, any]:
        """Get information about the sparsity structure."""
        info = {
            'n': self.n,
            'J_n': self.J_n,
            'K': self.K,
            'tau': self.tau,
            'weight_selector': type(self.weight_selector).__name__,
            'slab_distribution': type(self.slab_distribution).__name__,
            'weights': self.weights.copy(),
            'expected_nonzero_fraction': {}
        }
        
        # Compute expected fraction of non-zero coefficients at each level
        for j in range(self.J_n + 1):
            info['expected_nonzero_fraction'][j] = self.get_weight(j)
            
        return info
    
    def verify_boundedness_condition(self) -> bool:
        """Verify that the slab distribution satisfies the boundedness condition."""
        return self.slab_distribution.verify_boundedness(self.L0)
    
    def __repr__(self) -> str:
        return (f"SpikeSlabPrior(n={self.n}, J_n={self.J_n}, K={self.K}, "
                f"tau={self.tau}, weight_selector={type(self.weight_selector).__name__}, "
                f"slab_distribution={type(self.slab_distribution).__name__})")


 