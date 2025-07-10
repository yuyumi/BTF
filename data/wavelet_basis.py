import torch
import numpy as np
from typing import Tuple, Dict, List, Callable, Optional
import math


class HaarWaveletBasis:
    """
    Simple Haar wavelet basis implementation for [0,1].
    
    The Haar basis consists of:
    - Scaling function φ(x) = 1 for x ∈ [0,1], 0 otherwise
    - Wavelet function ψ(x) = 1 for x ∈ [0,0.5), -1 for x ∈ [0.5,1), 0 otherwise
    
    Basis functions:
    - Ψ_{j,k}(x) = 2^{j/2} ψ(2^j x - k) for j ≥ 0, k = 0, ..., 2^j - 1
    """
    
    def __init__(self, max_resolution: int = 8):
        """
        Initialize Haar wavelet basis.
        
        Args:
            max_resolution: Maximum resolution level J
        """
        self.max_resolution = max_resolution
        self.resolution_structure = self._compute_resolution_structure()
        
    def _compute_resolution_structure(self) -> Dict[int, int]:
        """Compute |I_j| for each resolution level j."""
        structure = {}
        
        # Level 0: scaling function (1 coefficient)
        structure[0] = 1
        
        # Levels 1 to max_resolution: wavelet functions
        for j in range(1, self.max_resolution + 1):
            structure[j] = 2 ** j
            
        return structure
    
    def scaling_function(self, x: torch.Tensor) -> torch.Tensor:
        """Haar scaling function φ(x)."""
        return torch.where((x >= 0) & (x < 1), 1.0, 0.0)
    
    def wavelet_function(self, x: torch.Tensor) -> torch.Tensor:
        """Haar wavelet function ψ(x)."""
        return torch.where(
            (x >= 0) & (x < 0.5), 1.0,
            torch.where((x >= 0.5) & (x < 1), -1.0, 0.0)
        )
    
    def basis_function(self, j: int, k: int, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Haar basis function Ψ_{j,k}(x).
        
        Args:
            j: Resolution level
            k: Translation parameter
            x: Input points
            
        Returns:
            Basis function values
        """
        if j == 0:
            # Scaling function
            return self.scaling_function(x)
        else:
            # Wavelet function: 2^{j/2} ψ(2^j x - k)
            scale = 2 ** (j / 2)
            shifted_x = 2 ** j * x - k
            return scale * self.wavelet_function(shifted_x)
    
    def compute_coefficient(self, func: Callable[[torch.Tensor], torch.Tensor], 
                          j: int, k: int, num_points: int = 1024) -> float:
        """
        Compute wavelet coefficient θ_{j,k} = ∫₀¹ f(x) Ψ_{j,k}(x) dx.
        
        Args:
            func: Function to analyze
            j: Resolution level
            k: Translation parameter
            num_points: Number of points for numerical integration
            
        Returns:
            Wavelet coefficient
        """
        x = torch.linspace(0, 1, num_points)
        dx = 1.0 / (num_points - 1)
        
        func_values = func(x)
        basis_values = self.basis_function(j, k, x)
        
        # Trapezoidal rule integration
        integrand = func_values * basis_values
        coefficient = torch.trapz(integrand, x).item()
        
        return coefficient
    
    def compute_all_coefficients(self, func: Callable[[torch.Tensor], torch.Tensor],
                               max_level: Optional[int] = None) -> Dict[Tuple[int, int], float]:
        """
        Compute all wavelet coefficients up to a given level.
        
        Args:
            func: Function to analyze
            max_level: Maximum resolution level (if None, uses max_resolution)
            
        Returns:
            Dictionary mapping (j,k) -> θ_{j,k}
        """
        if max_level is None:
            max_level = self.max_resolution
            
        coefficients = {}
        
        for j in range(max_level + 1):
            if j == 0:
                # Scaling coefficient
                coefficients[(0, 0)] = self.compute_coefficient(func, 0, 0)
            else:
                # Wavelet coefficients
                for k in range(2 ** j):
                    coefficients[(j, k)] = self.compute_coefficient(func, j, k)
                    
        return coefficients
    
    def reconstruct_function(self, coefficients: Dict[Tuple[int, int], float],
                           x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct function from wavelet coefficients.
        
        Args:
            coefficients: Dictionary mapping (j,k) -> θ_{j,k}
            x: Points where to evaluate reconstructed function
            
        Returns:
            Reconstructed function values
        """
        result = torch.zeros_like(x)
        
        for (j, k), coeff in coefficients.items():
            if coeff != 0:  # Skip zero coefficients
                basis_values = self.basis_function(j, k, x)
                result += coeff * basis_values
                
        return result
    
    def get_resolution_structure(self, max_level: Optional[int] = None) -> Dict[int, int]:
        """Get the resolution structure |I_j| for each level."""
        if max_level is None:
            max_level = self.max_resolution
            
        structure = {}
        for j in range(max_level + 1):
            if j == 0:
                structure[j] = 1  # One scaling coefficient
            else:
                structure[j] = 2 ** j  # 2^j wavelet coefficients
                
        return structure
    
    def total_coefficients(self, max_level: Optional[int] = None) -> int:
        """Get total number of coefficients up to max_level."""
        structure = self.get_resolution_structure(max_level)
        return sum(structure.values())


class WaveletDataGenerator:
    """
    Generates synthetic data following the spike-slab wavelet framework.
    
    Creates:
    1. True wavelet coefficients θ_{j,k} from spike-slab prior
    2. Noisy observations Y_{j,k} = θ_{j,k} + n^{-1/2} ε_{j,k}
    """
    
    def __init__(self, spike_slab_prior, wavelet_basis: HaarWaveletBasis):
        """
        Initialize data generator.
        
        Args:
            spike_slab_prior: SpikeSlabPrior instance
            wavelet_basis: HaarWaveletBasis instance
        """
        self.prior = spike_slab_prior
        self.basis = wavelet_basis
        
    def generate_true_coefficients(self) -> Dict[Tuple[int, int], float]:
        """Generate true wavelet coefficients θ_{j,k} from spike-slab prior."""
        # Get resolution structure compatible with prior
        max_level = min(self.prior.J_n, self.basis.max_resolution)
        resolution_structure = self.basis.get_resolution_structure(max_level)
        
        # Sample coefficients from prior
        coefficients = self.prior.sample_coefficients(resolution_structure)
        
        return coefficients
    
    def add_noise(self, true_coefficients: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """
        Add noise to create observations Y_{j,k} = θ_{j,k} + n^{-1/2} ε_{j,k}.
        
        Args:
            true_coefficients: True coefficients θ_{j,k}
            
        Returns:
            Noisy observations Y_{j,k}
        """
        noisy_coefficients = {}
        noise_std = 1.0 / math.sqrt(self.prior.n)
        
        for (j, k), theta_jk in true_coefficients.items():
            epsilon_jk = torch.randn(1).item()
            Y_jk = theta_jk + noise_std * epsilon_jk
            noisy_coefficients[(j, k)] = Y_jk
            
        return noisy_coefficients
    
    def generate_sample(self) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
        """
        Generate a complete sample (true coefficients, noisy observations).
        
        Returns:
            Tuple of (true_coefficients, noisy_observations)
        """
        true_coefficients = self.generate_true_coefficients()
        noisy_observations = self.add_noise(true_coefficients)
        
        return true_coefficients, noisy_observations
    
    def generate_function_sample(self, num_points: int = 256) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a function sample with reconstruction.
        
        Args:
            num_points: Number of points for function evaluation
            
        Returns:
            Tuple of (x_points, true_function, noisy_reconstruction)
        """
        true_coeffs, noisy_coeffs = self.generate_sample()
        
        x = torch.linspace(0, 1, num_points)
        true_function = self.basis.reconstruct_function(true_coeffs, x)
        noisy_reconstruction = self.basis.reconstruct_function(noisy_coeffs, x)
        
        return x, true_function, noisy_reconstruction


 