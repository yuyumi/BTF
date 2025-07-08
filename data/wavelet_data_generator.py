import torch
import numpy as np
from typing import Tuple, Dict, List, Optional
import math

try:
    # Try relative imports first (when used as module)
    from .spike_slab_prior import (
        SpikeSlabPrior, 
        WeightSelector, 
        SlabDistribution,
        GeometricMeanWeights,
        NormalSlab
    )
    from .wavelet_basis import HaarWaveletBasis, WaveletDataGenerator
    from .data_generator import Prior
except ImportError:
    # Fall back to absolute imports (when run as script)
    from spike_slab_prior import (
        SpikeSlabPrior, 
        WeightSelector, 
        SlabDistribution,
        GeometricMeanWeights,
        NormalSlab
    )
    from wavelet_basis import HaarWaveletBasis, WaveletDataGenerator
    from data_generator import Prior


class WaveletSpikeSlabPrior(Prior):
    """
    Spike-slab prior for wavelet coefficients that integrates with the transformer framework.
    
    This class implements the complete data generation process:
    1. Generate true wavelet coefficients θ_{j,k} from spike-slab prior
    2. Add noise to create observations Y_{j,k} = θ_{j,k} + n^{-1/2} ε_{j,k}
    3. Format data for transformer training
    """
    
    def __init__(self, 
                 n: int = 256,
                 K: float = 1.0,
                 tau: float = 1.0,
                 max_resolution: int = 6,
                 sequence_length: Optional[int] = None,
                 weight_selector: Optional[WeightSelector] = None,
                 slab_distribution: Optional[SlabDistribution] = None):
        """
        Initialize wavelet spike-slab prior.
        
        Args:
            n: Sample size (determines resolution cutoff and noise level)
            K: Lower bound parameter for mixture weights
            tau: Sparsity parameter  
            max_resolution: Maximum wavelet resolution level
            sequence_length: Fixed sequence length (if None, uses total coefficients)
            weight_selector: Strategy for computing mixture weights
            slab_distribution: Distribution for slab component g(x)
        """
        self.n = n
        self.K = K
        self.tau = tau
        self.max_resolution = max_resolution
        self.sequence_length = sequence_length
        
        # Default strategies if not provided
        self.weight_selector = weight_selector or GeometricMeanWeights()
        self.slab_distribution = slab_distribution or NormalSlab(std=1.0)
        
        # Create spike-slab prior and wavelet basis
        self.spike_slab_prior = SpikeSlabPrior(
            n=n, 
            K=K, 
            tau=tau, 
            L0=1.0,
            max_resolution=max_resolution,
            weight_selector=self.weight_selector,
            slab_distribution=self.slab_distribution
        )
        self.wavelet_basis = HaarWaveletBasis(max_resolution=max_resolution)
        self.data_generator = WaveletDataGenerator(self.spike_slab_prior, self.wavelet_basis)
        
        # Determine actual resolution levels and sequence structure
        self.effective_max_level = min(self.spike_slab_prior.J_n, max_resolution)
        self.resolution_structure = self.wavelet_basis.get_resolution_structure(self.effective_max_level)
        self.total_coeffs = sum(self.resolution_structure.values())
        
        # Set up coefficient ordering for sequence generation
        self._setup_coefficient_ordering()
        
        # Dimension for the framework
        self.dim = 1  # Each coefficient is a scalar
        
    def _setup_coefficient_ordering(self):
        """Setup consistent ordering of coefficients for sequence generation."""
        self.coeff_indices = []
        
        # Order: (j, k) in increasing order of j, then k
        for j in range(self.effective_max_level + 1):
            if j == 0:
                self.coeff_indices.append((0, 0))
            else:
                for k in range(2 ** (j - 1)):
                    self.coeff_indices.append((j, k))
                    
        self.index_to_coeff = {i: coeff for i, coeff in enumerate(self.coeff_indices)}
        self.coeff_to_index = {coeff: i for i, coeff in enumerate(self.coeff_indices)}
        
    def sample(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Generate samples from the wavelet spike-slab prior.
        
        Args:
            batch_size: Number of sequences to generate
            seq_len: Length of each sequence
            
        Returns:
            Tensor of shape (batch_size, seq_len, 1) containing noisy observations Y_{j,k}
        """
        sequences = []
        
        for _ in range(batch_size):
            # Generate true coefficients and noisy observations
            true_coeffs, noisy_coeffs = self.data_generator.generate_sample()
            
            # Convert to sequence format
            sequence = self._coefficients_to_sequence(noisy_coeffs, seq_len)
            sequences.append(sequence)
            
        return torch.stack(sequences)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of noisy observations.
        
        This is complex for the spike-slab model as it requires marginalizing
        over the true coefficients. For training, we approximate this.
        
        Args:
            x: Noisy observations of shape (batch_size, seq_len, 1)
            
        Returns:
            Log probabilities of shape (batch_size, seq_len, 1)
        """
        batch_size, seq_len, _ = x.shape
        log_probs = torch.zeros_like(x)
        
        # Noise standard deviation
        noise_std = 1.0 / math.sqrt(self.n)
        
        # For each position in sequence, compute approximate log probability
        for i in range(min(seq_len, len(self.coeff_indices))):
            j, k = self.coeff_indices[i]
            w_jn = self.spike_slab_prior.get_weight(j)
            
            if w_jn == 0.0:
                # Pure spike: Y_{j,k} ~ N(0, n^{-1/2})
                log_probs[:, i, 0] = -0.5 * (x[:, i, 0] ** 2) / (noise_std ** 2) - 0.5 * math.log(2 * math.pi * noise_std ** 2)
            else:
                # Mixture: approximate as mixture of normals
                # Spike component: Y_{j,k} ~ N(0, n^{-1/2})  
                # Slab component: Y_{j,k} ~ N(0, slab_var + n^{-1/2})
                
                spike_var = noise_std ** 2
                
                # Get slab variance from the distribution
                if hasattr(self.slab_distribution, 'std'):
                    # Normal distribution
                    slab_var = self.slab_distribution.std ** 2 + noise_std ** 2
                elif hasattr(self.slab_distribution, 'scale'):
                    # Laplace distribution: variance = 2 * scale^2
                    slab_var = 2 * (self.slab_distribution.scale ** 2) + noise_std ** 2
                elif hasattr(self.slab_distribution, 'half_width'):
                    # Uniform distribution: variance = (2*half_width)^2 / 12
                    slab_var = (2 * self.slab_distribution.half_width) ** 2 / 12 + noise_std ** 2
                else:
                    # Default fallback
                    slab_var = 1.0 + noise_std ** 2
                
                spike_log_prob = -0.5 * (x[:, i, 0] ** 2) / spike_var - 0.5 * math.log(2 * math.pi * spike_var)
                slab_log_prob = -0.5 * (x[:, i, 0] ** 2) / slab_var - 0.5 * math.log(2 * math.pi * slab_var)
                
                # Log-sum-exp of mixture
                log_probs[:, i, 0] = torch.logsumexp(
                    torch.stack([
                        math.log(1 - w_jn) + spike_log_prob,
                        math.log(w_jn) + slab_log_prob
                    ], dim=0), dim=0
                )
                
        return log_probs
    
    def _coefficients_to_sequence(self, coefficients: Dict[Tuple[int, int], float], seq_len: int) -> torch.Tensor:
        """Convert coefficient dictionary to sequence tensor."""
        if self.sequence_length is not None:
            seq_len = self.sequence_length
            
        sequence = torch.zeros(seq_len, 1)
        
        for i in range(min(seq_len, len(self.coeff_indices))):
            j, k = self.coeff_indices[i]
            if (j, k) in coefficients:
                sequence[i, 0] = coefficients[(j, k)]
                
        return sequence
    
    def generate_training_batch(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """
        Generate training batch with both inputs (noisy) and targets (true).
        
        Args:
            batch_size: Number of samples
            seq_len: Sequence length
            
        Returns:
            Dictionary with 'x' (noisy observations) and 'log_prob' (true coefficients)
        """
        noisy_sequences = []
        true_sequences = []
        
        for _ in range(batch_size):
            true_coeffs, noisy_coeffs = self.data_generator.generate_sample()
            
            noisy_seq = self._coefficients_to_sequence(noisy_coeffs, seq_len)
            true_seq = self._coefficients_to_sequence(true_coeffs, seq_len)
            
            noisy_sequences.append(noisy_seq)
            true_sequences.append(true_seq)
            
        return {
            'x': torch.stack(noisy_sequences),           # Input: noisy observations Y_{j,k}
            'log_prob': torch.stack(true_sequences),     # Target: true coefficients θ_{j,k}
        }
    
    def reconstruct_function(self, coefficients: torch.Tensor, num_points: int = 256) -> torch.Tensor:
        """
        Reconstruct function from coefficient sequence.
        
        Args:
            coefficients: Coefficient sequence of shape (seq_len, 1)
            num_points: Number of evaluation points
            
        Returns:
            Function values at evaluation points
        """
        # Convert sequence back to coefficient dictionary
        coeff_dict = {}
        seq_len = coefficients.shape[0]
        
        for i in range(min(seq_len, len(self.coeff_indices))):
            j, k = self.coeff_indices[i]
            coeff_dict[(j, k)] = coefficients[i, 0].item()
            
        # Reconstruct using wavelet basis
        x = torch.linspace(0, 1, num_points)
        return self.wavelet_basis.reconstruct_function(coeff_dict, x)
    
    def get_info(self) -> Dict[str, any]:
        """Get information about the data generation setup."""
        return {
            'n': self.n,
            'K': self.K,
            'tau': self.tau,
            'max_resolution': self.max_resolution,
            'effective_max_level': self.effective_max_level,
            'J_n': self.spike_slab_prior.J_n,
            'total_coefficients': self.total_coeffs,
            'sequence_length': self.sequence_length or self.total_coeffs,
            'resolution_structure': self.resolution_structure,
            'noise_std': 1.0 / math.sqrt(self.n),
            'weight_selector': type(self.weight_selector).__name__,
            'slab_distribution': type(self.slab_distribution).__name__,
            'sparsity_weights': self.spike_slab_prior.weights
        }


# Factory function for easy creation with different configurations
def create_wavelet_prior(config_name: str, n: int = 256, **kwargs) -> WaveletSpikeSlabPrior:
    """
    Factory function to create pre-configured wavelet priors.
    
    Args:
        config_name: Name of configuration ('standard', 'laplace', 'uniform_weights', etc.)
        n: Sample size
        **kwargs: Additional parameters
        
    Returns:
        Configured WaveletSpikeSlabPrior
    """
    try:
        from .spike_slab_prior import (
            UniformWeights, ArithmeticMeanWeights, ExponentialDecayWeights,
            LaplaceSlab, UniformSlab, StudentTSlab
        )
    except ImportError:
        from spike_slab_prior import (
            UniformWeights, ArithmeticMeanWeights, ExponentialDecayWeights,
            LaplaceSlab, UniformSlab, StudentTSlab
        )
    
    configs = {
        'standard': {
            'weight_selector': GeometricMeanWeights(),
            'slab_distribution': NormalSlab(std=1.0)
        },
        'laplace': {
            'weight_selector': GeometricMeanWeights(),
            'slab_distribution': LaplaceSlab(scale=0.7)
        },
        'uniform_weights': {
            'weight_selector': UniformWeights(seed=42),
            'slab_distribution': NormalSlab(std=1.0)
        },
        'heavy_tails': {
            'weight_selector': GeometricMeanWeights(),
            'slab_distribution': StudentTSlab(df=3.0, scale=1.0)
        },
        'uniform_slab': {
            'weight_selector': ArithmeticMeanWeights(),
            'slab_distribution': UniformSlab(half_width=2.0)
        },
        'exponential_decay': {
            'weight_selector': ExponentialDecayWeights(w0=0.3, alpha=0.7),
            'slab_distribution': NormalSlab(std=1.2)
        }
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config '{config_name}'. Available: {list(configs.keys())}")
    
    config = configs[config_name]
    config.update(kwargs)  # Allow overrides
    
    return WaveletSpikeSlabPrior(n=n, **config)


 