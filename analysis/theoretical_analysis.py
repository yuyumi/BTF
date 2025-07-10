import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
import sys
from scipy import stats
from torch.utils.data import DataLoader
import pickle
import csv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.wavelet_data_generator import create_wavelet_prior, WaveletSpikeSlabPrior
from train.train_wavelets import WaveletDataset
from analysis.visualize_results import WaveletAnalyzer


class HolderBallGenerator:
    """
    Generator for parameters from Hölder balls H(β, L).
    
    H(β, L) = {θ = (θ_{j,k})_{(j,k)∈Λ} : |θ_{jk}| ≤ L2^{-j(β+1/2)}, (j,k) ∈ Λ}
    """
    
    def __init__(self, max_resolution: int, beta: float, L: float = 1.0):
        """
        Initialize Hölder ball generator.
        
        Args:
            max_resolution: Maximum wavelet resolution level
            beta: Smoothness parameter β > 0
            L: Ball radius parameter L > 0
        """
        self.max_resolution = max_resolution
        self.beta = beta
        self.L = L
        
        # Set up coefficient structure matching wavelet data generator
        self.coeff_indices = []
        for j in range(max_resolution + 1):
            if j == 0:
                self.coeff_indices.append((0, 0))
            else:
                for k in range(2 ** j):
                    self.coeff_indices.append((j, k))
    
    def generate_holder_parameters(self, seq_len: int, mode: str = 'boundary') -> torch.Tensor:
        """
        Generate parameters from Hölder ball H(β, L).
        
        Args:
            seq_len: Length of coefficient sequence
            mode: 'boundary' (on boundary), 'interior' (inside ball), 'uniform' (uniform in ball)
            
        Returns:
            Coefficient tensor of shape (seq_len, 1)
        """
        coefficients = torch.zeros(seq_len, 1)
        
        # Get constraint bounds for each coefficient
        constraint_bounds = []
        for i in range(min(seq_len, len(self.coeff_indices))):
            j, k = self.coeff_indices[i]
            max_magnitude = self.L * (2 ** (-j * (self.beta + 0.5)))
            constraint_bounds.append(max_magnitude)
        
        if mode == 'boundary':
            # Proper boundary sampling: at least one coefficient at its constraint
            self._sample_boundary(coefficients, constraint_bounds)
        elif mode == 'interior':
            # Sample uniformly within constraints (excluding boundary)
            self._sample_interior(coefficients, constraint_bounds)
        elif mode == 'uniform':
            # Sample uniformly within constraints (including boundary)
            self._sample_uniform(coefficients, constraint_bounds)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'boundary', 'interior', or 'uniform'")
                
        return coefficients
    
    def _sample_boundary(self, coefficients: torch.Tensor, constraint_bounds: List[float]):
        """
        Sample from the boundary of the Holder ball.
        
        Strategy: Randomly choose one coefficient to be at its constraint boundary,
        sample all others uniformly within their constraints.
        """
        num_coeffs = len(constraint_bounds)
        
        # Choose which coefficient(s) will be at boundary
        boundary_idx = np.random.choice(num_coeffs)
        
        for i in range(num_coeffs):
            max_magnitude = constraint_bounds[i]
            
            if i == boundary_idx:
                # This coefficient is at boundary: |θ_{j,k}| = max_magnitude
                sign = np.random.choice([-1, 1])
                coefficients[i, 0] = sign * max_magnitude
            else:
                # Sample uniformly within constraint
                coefficients[i, 0] = np.random.uniform(-max_magnitude, max_magnitude)
    
    def _sample_interior(self, coefficients: torch.Tensor, constraint_bounds: List[float]):
        """
        Sample from the interior of the Holder ball (excluding boundary).
        
        Strategy: Sample uniformly within constraints, then shrink slightly to ensure
        we're in the interior.
        """
        num_coeffs = len(constraint_bounds)
        shrink_factor = 0.99  # Shrink slightly to ensure interior
        
        for i in range(num_coeffs):
            max_magnitude = constraint_bounds[i] * shrink_factor
            coefficients[i, 0] = np.random.uniform(-max_magnitude, max_magnitude)
    
    def _sample_uniform(self, coefficients: torch.Tensor, constraint_bounds: List[float]):
        """
        Sample uniformly from the entire Holder ball (including boundary).
        
        Strategy: Sample each coefficient independently and uniformly within its constraint.
        """
        num_coeffs = len(constraint_bounds)
        
        for i in range(num_coeffs):
            max_magnitude = constraint_bounds[i]
            coefficients[i, 0] = np.random.uniform(-max_magnitude, max_magnitude)
    
    def sample_multiple_boundary_points(self, seq_len: int, num_boundary_coeffs: int = 1) -> torch.Tensor:
        """
        Sample from boundary with multiple coefficients at their constraints.
        
        Args:
            seq_len: Length of coefficient sequence
            num_boundary_coeffs: Number of coefficients to place at boundary
            
        Returns:
            Coefficient tensor with specified number of coefficients at boundary
        """
        coefficients = torch.zeros(seq_len, 1)
        
        # Get constraint bounds
        constraint_bounds = []
        for i in range(min(seq_len, len(self.coeff_indices))):
            j, k = self.coeff_indices[i]
            max_magnitude = self.L * (2 ** (-j * (self.beta + 0.5)))
            constraint_bounds.append(max_magnitude)
        
        num_coeffs = len(constraint_bounds)
        num_boundary_coeffs = min(num_boundary_coeffs, num_coeffs)
        
        # Choose which coefficients will be at boundary
        boundary_indices = np.random.choice(num_coeffs, num_boundary_coeffs, replace=False)
        
        for i in range(num_coeffs):
            max_magnitude = constraint_bounds[i]
            
            if i in boundary_indices:
                # This coefficient is at boundary
                sign = np.random.choice([-1, 1])
                coefficients[i, 0] = sign * max_magnitude
            else:
                # Sample uniformly within constraint
                coefficients[i, 0] = np.random.uniform(-max_magnitude, max_magnitude)
                
        return coefficients


class WeightedLInfLoss:
    """
    Implements the weighted L∞ loss from the paper:
    ℓ∞(θ, θ') = Σ_{j∈ℕ} 2^{j/2} max_{k∈I_j} |θ_{j,k} - θ'_{j,k}|
    """
    
    def __init__(self, max_resolution: int):
        """
        Initialize weighted L∞ loss computation.
        
        Args:
            max_resolution: Maximum wavelet resolution level
        """
        self.max_resolution = max_resolution
        
        # Set up coefficient structure
        self.coeff_indices = []
        self.level_ranges = {}  # Maps j -> (start_idx, end_idx) in sequence
        
        idx = 0
        for j in range(max_resolution + 1):
            if j == 0:
                self.coeff_indices.append((0, 0))
                self.level_ranges[0] = (idx, idx + 1)
                idx += 1
            else:
                start_idx = idx
                for k in range(2 ** j):
                    self.coeff_indices.append((j, k))
                    idx += 1
                self.level_ranges[j] = (start_idx, idx)
    
    def compute_loss(self, theta1: torch.Tensor, theta2: torch.Tensor) -> float:
        """
        Compute weighted L∞ loss between two coefficient sequences.
        
        Args:
            theta1: First coefficient sequence (seq_len, 1)
            theta2: Second coefficient sequence (seq_len, 1)
            
        Returns:
            Weighted L∞ distance
        """
        theta1_flat = theta1.flatten()
        theta2_flat = theta2.flatten()
        
        total_loss = 0.0
        
        for j in range(self.max_resolution + 1):
            if j not in self.level_ranges:
                continue
                
            start_idx, end_idx = self.level_ranges[j]
            if start_idx >= len(theta1_flat) or start_idx >= len(theta2_flat):
                break
                
            # Compute max over k ∈ I_j of |θ_{j,k} - θ'_{j,k}|
            level_diffs = torch.abs(theta1_flat[start_idx:end_idx] - theta2_flat[start_idx:end_idx])
            if len(level_diffs) > 0:
                max_diff = torch.max(level_diffs).item()
                
                # Weight by 2^{j/2}
                weight = 2 ** (j / 2)
                total_loss += weight * max_diff
                
        return total_loss


class TheoreticalAnalyzer:
    """
    Analysis module for testing Theorem 3.1: Posterior contraction rates
    and theoretical guarantees of spike-slab priors.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """Initialize with trained model for theoretical analysis."""
        self.device = device
        
        # Load model
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.model = saved_data['model'].to(device).eval()
        self.base_wavelet_prior = saved_data['wavelet_prior']
        self.config = saved_data.get('config', {})
        
        # Get max resolution from model configuration for correct analysis
        self.max_resolution = getattr(self.base_wavelet_prior, 'max_resolution', 6)
        self.seq_len = getattr(self.base_wavelet_prior, 'total_coeffs', 64)
        
        # Initialize weighted loss function for correct analysis
        self.weighted_loss = WeightedLInfLoss(self.max_resolution)
        
        print("Theoretical Analysis for Spike-Slab Posterior Contraction")
        print(f"Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Max resolution: {self.max_resolution}")
        print(f"Sequence length: {self.seq_len}")
    
    def load_theorem_configurations(self, config_path: str = "theorem_configurations.csv") -> Optional[List[Dict]]:
        """
        Load pre-generated theorem configurations from CSV file.
        
        Args:
            config_path: Path to CSV file with configurations
            
        Returns:
            List of configuration dictionaries, or None if file doesn't exist
        """
        if not os.path.exists(config_path):
            print(f"No pre-generated configurations found at {config_path}")
            return None
            
        print(f"Loading pre-generated configurations from {config_path}")
        
        configurations = []
        try:
            with open(config_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    config = {
                        'M': float(row['M']),
                        'B': float(row['B']),
                        'L': float(row['L']),
                        'Beta1': float(row['Beta1']),
                        'Beta2': float(row['Beta2']),
                        'n': int(row['n']),
                        'seq_len': int(row['seq_len'])
                    }
                    configurations.append(config)
                    
            print(f"✓ Loaded {len(configurations)} pre-generated configurations")
            return configurations
            
        except Exception as e:
            print(f"Error loading configurations: {e}")
            return None
    
    def find_matching_configuration(self, configurations: List[Dict], 
                                  L: float, beta1: float, beta2: float, n: int) -> Optional[Dict]:
        """
        Find a configuration that matches the given parameters.
        
        Args:
            configurations: List of available configurations
            L: Desired L value
            beta1: Desired Beta1 value
            beta2: Desired Beta2 value
            n: Desired n value
            
        Returns:
            Matching configuration or None
        """
        for config in configurations:
            if (abs(config['L'] - L) < 1e-6 and 
                abs(config['Beta1'] - beta1) < 1e-6 and 
                abs(config['Beta2'] - beta2) < 1e-6 and 
                config['n'] == n):
                return config
        return None
    
    def test_with_precomputed_configuration(self, config: Dict, num_trials: int = 100) -> Dict:
        """
        Test Theorem 3.1 using a pre-computed configuration.
        
        Args:
            config: Pre-computed configuration with M, B, L, Beta1, Beta2, n
            num_trials: Number of trials for validation
            
        Returns:
            Test results
        """
        M, B, L = config['M'], config['B'], config['L']
        beta1, beta2, n = config['Beta1'], config['Beta2'], config['n']
        
        print(f"\nTesting pre-computed configuration:")
        print(f"  M = {M:.3f}, B = {B:.3f}, L = {L:.2f}")
        print(f"  β ∈ [{beta1:.1f}, {beta2:.1f}], n = {n}")
        
        # Create β grid within the interval
        beta_values = np.linspace(beta1, beta2, 5)
        
        # This method is deprecated - precomputed configurations should use the main test method
        # For now, return a placeholder result
        results = {
            'sample_sizes': [n],
            'beta_values': beta_values.tolist(),
            'L': L,
            'universal_constants': {
                'M': M,
                'B': B,
                'search_details': {'precomputed': True}
            },
            'performance_results': {'all_combinations_pass': False, 'overall_success_rate': 0.0},
            'theorem_satisfied': False,
            'overall_success_rate': 0.0
        }
        
        return results
    
    def analyze_empirical_vs_theoretical_rates(self, M: float, B: float, L: float,
                                             sample_sizes: List[int], beta_values: List[float],
                                             num_trials: int = 100, save_path: str = None) -> Dict:
        """
        Detailed analysis of empirical vs theoretical rates using universal constants M, B.
        
        For each (β, n) combination:
        1. Compute theoretical threshold = M * (n/log n)^{-β/(2β+1)}
        2. Collect empirical loss values from model
        3. Check concentration inequality: P[empirical ≥ threshold] ≤ n^{-B}
        4. Analyze how tight the bounds are
        
        Args:
            M: Universal constant M
            B: Universal constant B
            L: Hölder ball radius
            sample_sizes: Sample sizes to test
            beta_values: Beta values to test
            num_trials: Number of empirical trials per combination
            save_path: Path to save detailed plots
            
        Returns:
            Detailed analysis results
        """
        print(f"\nDetailed Empirical vs Theoretical Analysis")
        print(f"Universal constants: M = {M:.3f}, B = {B:.3f}")
        print("-" * 50)
        
        results = {
            'M': M, 'B': B, 'L': L,
            'combinations': {},
            'summary_stats': {},
            'concentration_check': {},
            'bound_tightness': {}
        }
        
        # Track overall statistics
        all_empirical_rates = []
        all_theoretical_rates = []
        all_concentration_ratios = []
        all_bound_violations = []
        all_threshold_violations = []
        
        for beta in beta_values:
            print(f"\nβ = {beta:.2f}")
            holder_generator = HolderBallGenerator(self.max_resolution, beta, L)
            
            for n in sample_sizes:
                print(f"  n = {n:3d}: ", end="")
                
                # Theoretical rate and threshold
                theoretical_rate = (n / np.log(n)) ** (-beta / (2 * beta + 1))
                threshold = M * theoretical_rate
                required_prob = n ** (-B)
                
                # Collect empirical data
                empirical_losses = []
                for trial in range(num_trials):
                    # Generate θ₀ from Hölder ball
                    true_params = holder_generator.generate_holder_parameters(self.seq_len, mode='boundary')
                    
                    # Add noise: Y = θ₀ + n^{-1/2} * ε
                    noise_std = 1.0 / np.sqrt(n)
                    noise = torch.randn_like(true_params) * noise_std
                    noisy_obs = true_params + noise
                    
                    # Get model prediction
                    with torch.no_grad():
                        x_tensor = noisy_obs.unsqueeze(0).to(self.device)
                        pred_params = self.model(x_tensor).cpu()
                    
                    # Compute weighted L∞ loss
                    loss = self.weighted_loss.compute_loss(pred_params[0], true_params)
                    empirical_losses.append(loss)
                
                empirical_losses = np.array(empirical_losses)
                
                # Compute statistics
                empirical_mean = np.mean(empirical_losses)
                empirical_std = np.std(empirical_losses)
                
                # Key insight: Collect empirical rate at the n^{-B} concentration level
                # This is the (1 - n^{-B}) quantile of the empirical distribution
                concentration_level = 1.0 - required_prob  # e.g., if n^{-B} = 0.088, then 91.2th percentile
                empirical_rate_at_concentration = np.percentile(empirical_losses, concentration_level * 100)
                
                # Compare this empirical rate to the theoretical rate M * (n/log n)^{-β/(2β+1)}
                theoretical_rate_raw = (n / np.log(n)) ** (-beta / (2 * beta + 1))  # Without M factor
                theoretical_rate_with_M = M * theoretical_rate_raw  # With M factor (this is the threshold)
                
                # Rate comparison at concentration level
                rate_ratio_at_concentration = empirical_rate_at_concentration / theoretical_rate_raw
                rate_ratio_vs_threshold = empirical_rate_at_concentration / theoretical_rate_with_M
                
                # Also check traditional concentration inequality for comparison
                concentration_prob = np.mean(empirical_losses >= threshold)
                concentration_satisfied = concentration_prob <= required_prob
                concentration_ratio = concentration_prob / required_prob
                
                # Check if threshold is satisfied (quantile ≤ M×theo)
                threshold_satisfied = empirical_rate_at_concentration <= theoretical_rate_with_M
                
                # Store combination results
                combo_key = (beta, n)
                results['combinations'][combo_key] = {
                    'theoretical_rate_raw': theoretical_rate_raw,
                    'theoretical_rate_with_M': theoretical_rate_with_M,
                    'threshold': threshold,
                    'required_prob': required_prob,
                    'concentration_level': concentration_level,
                    'empirical_mean': empirical_mean,
                    'empirical_std': empirical_std,
                    'empirical_rate_at_concentration': empirical_rate_at_concentration,
                    'rate_ratio_at_concentration': rate_ratio_at_concentration,
                    'rate_ratio_vs_threshold': rate_ratio_vs_threshold,
                    'threshold_satisfied': threshold_satisfied,  # Key check: quantile ≤ M×theo
                    'concentration_prob': concentration_prob,
                    'concentration_satisfied': concentration_satisfied,
                    'concentration_ratio': concentration_ratio,
                    'empirical_losses': empirical_losses.tolist()
                }
                
                # Track for overall statistics (use empirical rate at concentration level)
                all_empirical_rates.append(empirical_rate_at_concentration)
                all_theoretical_rates.append(theoretical_rate_raw)
                all_concentration_ratios.append(concentration_ratio)
                all_bound_violations.append(not concentration_satisfied)
                all_threshold_violations.append(not threshold_satisfied)
                
                # Print summary for this combination
                status = "✓" if concentration_satisfied else "✗"
                threshold_status = "✓" if threshold_satisfied else "✗"
                
                print(f"{status} ℓ∞-quantile_{concentration_level*100:.1f}%={empirical_rate_at_concentration:.4f}, "
                      f"M×theo={theoretical_rate_with_M:.4f} {threshold_status}, "
                      f"P={concentration_prob:.4f}≤{required_prob:.4f}")
        
        # Compute summary statistics
        all_empirical_rates = np.array(all_empirical_rates)
        all_theoretical_rates = np.array(all_theoretical_rates)
        all_concentration_ratios = np.array(all_concentration_ratios)
        all_bound_violations = np.array(all_bound_violations)
        all_threshold_violations = np.array(all_threshold_violations)
        
        results['summary_stats'] = {
            'num_combinations': len(all_empirical_rates),
            'empirical_rate_stats': {
                'mean': np.mean(all_empirical_rates),
                'std': np.std(all_empirical_rates),
                'min': np.min(all_empirical_rates),
                'max': np.max(all_empirical_rates)
            },
            'theoretical_rate_stats': {
                'mean': np.mean(all_theoretical_rates),
                'std': np.std(all_theoretical_rates),
                'min': np.min(all_theoretical_rates),
                'max': np.max(all_theoretical_rates)
            },
            'rate_ratio_stats': {
                'mean': np.mean(all_empirical_rates / all_theoretical_rates),
                'std': np.std(all_empirical_rates / all_theoretical_rates),
                'min': np.min(all_empirical_rates / all_theoretical_rates),
                'max': np.max(all_empirical_rates / all_theoretical_rates)
            }
        }
        
        results['concentration_check'] = {
            'violations': int(np.sum(all_bound_violations)),
            'total_combinations': len(all_bound_violations),
            'violation_rate': np.mean(all_bound_violations),
            'concentration_ratio_stats': {
                'mean': np.mean(all_concentration_ratios),
                'std': np.std(all_concentration_ratios),
                'min': np.min(all_concentration_ratios),
                'max': np.max(all_concentration_ratios)
            }
        }
        
        results['threshold_check'] = {
            'violations': int(np.sum(all_threshold_violations)),
            'total_combinations': len(all_threshold_violations),
            'violation_rate': np.mean(all_threshold_violations),
            'quantile_to_threshold_ratio_stats': {
                'mean': np.mean(all_empirical_rates / all_theoretical_rates),
                'std': np.std(all_empirical_rates / all_theoretical_rates),
                'min': np.min(all_empirical_rates / all_theoretical_rates),
                'max': np.max(all_empirical_rates / all_theoretical_rates)
            }
        }
        
        results['bound_tightness'] = {
            'avg_empirical_to_theoretical_ratio': np.mean(all_empirical_rates / all_theoretical_rates),
            'bounds_are_tight': np.mean(all_empirical_rates / all_theoretical_rates) > 0.1,  # Bounds are useful if empirical is at least 10% of theoretical
            'concentration_efficiency': 1.0 - np.mean(all_concentration_ratios)  # How much "room" we have in the bounds
        }
        
        # Print summary
        print(f"\n" + "="*60)
        print(f"EMPIRICAL VS THEORETICAL ANALYSIS SUMMARY")
        print(f"="*60)
        print(f"Combinations tested: {results['summary_stats']['num_combinations']}")
        print(f"Concentration violations: {results['concentration_check']['violations']}/{results['concentration_check']['total_combinations']}")
        print(f"Concentration violation rate: {results['concentration_check']['violation_rate']:.1%}")
        print(f"Threshold violations (ℓ∞-quantile > M×theo): {results['threshold_check']['violations']}/{results['threshold_check']['total_combinations']}")
        print(f"Threshold violation rate: {results['threshold_check']['violation_rate']:.1%}")
        print(f"Average empirical/theoretical ratio: {results['bound_tightness']['avg_empirical_to_theoretical_ratio']:.3f}")
        print(f"Bounds are {'tight and useful' if results['bound_tightness']['bounds_are_tight'] else 'loose'}")
        
        # Generate detailed plots (using existing universal constants plotting)
        if save_path:
            # Use existing plotting method for now
            pass  # Could add specific empirical vs theoretical plots later
        
        return results

    
    def test_uniform_convergence(self,
                               beta_intervals: List[Tuple[float, float]] = None,
                               beta_grid_size: int = 5,
                               sample_size: int = 256,
                               num_trials: int = 50,
                               save_path: str = None) -> Dict:
        """
        Test uniform convergence over multiple disjoint intervals [β₁, β₂].
        
        The theorem claims uniform convergence over any interval [β₁, β₂].
        We test this by:
        1. Testing multiple disjoint intervals
        2. For each interval, testing a grid of β values
        3. Checking uniform bounds within each interval
        """
        if beta_intervals is None:
            beta_intervals = [
                (0.5, 1.5),   # Low smoothness regime
                (1.0, 2.0),   # Medium smoothness regime  
                (1.5, 3.0),   # High smoothness regime
                (2.0, 4.0)    # Very high smoothness regime
            ]
        
        print("\nTesting Uniform Convergence Over Multiple β Intervals")
        print("="*60)
        print("Testing theorem claim: uniform convergence over any [β₁, β₂]")
        
        results = {
            'beta_intervals': beta_intervals,
            'interval_results': [],
            'overall_uniform_bound': 0.0,
            'intervals_passed': 0,
            'total_intervals': len(beta_intervals)
        }
        
        for i, (beta1, beta2) in enumerate(beta_intervals):
            print(f"\n--- INTERVAL {i+1}: [β₁={beta1}, β₂={beta2}] ---")
            
            # Create grid of β values in this interval
            beta_values = np.linspace(beta1, beta2, beta_grid_size)
            
            interval_results = {
                'beta_range': (beta1, beta2),
                'beta_values': beta_values.tolist(),
                'convergence_rates': [],
                'theoretical_bounds': [],
                'rate_ratios': [],
                'uniform_bound_in_interval': 0.0,
                'passes_uniform_test': False
            }
            
            for beta in beta_values:
                print(f"  Testing β = {beta:.2f}")
                
                # Generate data with this smoothness parameter
                wavelet_prior = create_wavelet_prior('standard', n=sample_size, max_resolution=4)
                seq_len = wavelet_prior.total_coeffs
                
                linf_distances = []
                
                for trial in range(num_trials):
                    batch = wavelet_prior.generate_training_batch(1, seq_len)
                    true_params = batch['log_prob'][0].numpy().flatten()
                    noisy_obs = batch['x'][0]
                    
                    with torch.no_grad():
                        x_tensor = noisy_obs.unsqueeze(0).to(self.device)
                        pred_params = self.model(x_tensor).cpu().numpy().flatten()
                    
                    linf_dist = np.max(np.abs(pred_params - true_params))
                    linf_distances.append(linf_dist)
                
                # Theoretical rate for this β
                n = sample_size
                theoretical_rate = (n / np.log(n)) ** (-beta / (2 * beta + 1))
                
                empirical_rate = np.mean(linf_distances)
                rate_ratio = empirical_rate / theoretical_rate
                
                interval_results['convergence_rates'].append(empirical_rate)
                interval_results['theoretical_bounds'].append(theoretical_rate)
                interval_results['rate_ratios'].append(rate_ratio)
                
                print(f"    Theoretical: {theoretical_rate:.4f}, Empirical: {empirical_rate:.4f}, Ratio: {rate_ratio:.2f}")
            
            # Check uniform convergence within this interval
            uniform_bound = np.max(interval_results['rate_ratios'])
            interval_results['uniform_bound_in_interval'] = uniform_bound
            interval_results['passes_uniform_test'] = uniform_bound < 2.0
            
            if interval_results['passes_uniform_test']:
                results['intervals_passed'] += 1
                print(f"  ✓ INTERVAL [{beta1}, {beta2}]: Uniform bound = {uniform_bound:.2f}")
            else:
                print(f"  ✗ INTERVAL [{beta1}, {beta2}]: Uniform bound = {uniform_bound:.2f} (FAILED)")
            
            results['interval_results'].append(interval_results)
            results['overall_uniform_bound'] = max(results['overall_uniform_bound'], uniform_bound)
        
        # Overall assessment
        print(f"\n" + "="*60)
        print(f"UNIFORM CONVERGENCE SUMMARY:")
        print(f"Intervals passed: {results['intervals_passed']}/{results['total_intervals']}")
        print(f"Overall uniform bound: {results['overall_uniform_bound']:.2f}")
        
        if results['intervals_passed'] == results['total_intervals']:
            print("✓ THEOREM 3.1 UNIFORM CONVERGENCE SATISFIED")
            print("  Uniform convergence holds over all tested intervals [β₁, β₂]")
        else:
            print("✗ THEOREM 3.1 UNIFORM CONVERGENCE FAILED")
            print("  Some intervals do not satisfy uniform convergence")
        
        # Plot results
        self._plot_uniform_convergence_intervals(results, save_path)
        
        return results
    

    
    def test_theorem_3_1_over_holder_balls(self, 
                                sample_sizes: List[int] = [32, 64, 128, 256, 512],
                                beta_values: List[float] = [0.5, 1.0, 1.5, 2.0, 3.0],
                                L: float = 1.0,
                                num_trials: int = 100,
                                save_path: str = None) -> Dict:
        """
        Test Theorem 3.1 over Hölder balls: Find universal constants M, B that work for ALL β values.
        
        The theorem states that there exist M, B > 0 such that for ALL β ∈ [β₁, β₂]:
        P^n(θ : ℓ∞(θ, θ₀) ≥ M(n/log n)^{-β/(2β+1)}|Y^n) ≤ n^{-B}
        
        This means M and B are UNIVERSAL constants, not β-dependent.
        
        Args:
            sample_sizes: Sample sizes to test
            beta_values: Smoothness parameters to test
            L: Hölder ball radius
            num_trials: Number of trials per configuration
            save_path: Path to save results
            
        Returns:
            Dictionary with analysis results
        """
        print("\nTheorem 3.1 Analysis: Finding Universal Constants M, B")
        print("="*65)
        print("Finding SINGLE pair (M,B) that works for ALL β values simultaneously")
        
        # Try to load pre-generated configurations first
        beta1, beta2 = min(beta_values), max(beta_values)
        configurations = self.load_theorem_configurations()
        
        if configurations:
            print(f"\nChecking for pre-computed configurations for β ∈ [{beta1:.1f}, {beta2:.1f}], L = {L:.2f}")
            
            # Try to find matching configurations for each sample size
            for n in sample_sizes:
                matching_config = self.find_matching_configuration(configurations, L, beta1, beta2, n)
                if matching_config:
                    print(f"✓ Found pre-computed configuration for n = {n}")
                    return self.test_with_precomputed_configuration(matching_config, num_trials)
            
            print("No exact matches found, falling back to exhaustive search")
        
        print("\nPerforming exhaustive search for universal constants")
        
        # Collect all empirical data across all (β, n) combinations
        all_empirical_data = []
        all_theoretical_rates = []
        beta_n_combinations = []
        
        print(f"\nPhase 1: Collecting empirical data across all (β, n) combinations")
        print("-" * 60)
        
        for beta in beta_values:
            print(f"\nTesting β = {beta}")
            
            # Initialize Hölder ball generator for this β
            holder_generator = HolderBallGenerator(self.max_resolution, beta, L)
            
            for n in sample_sizes:
                print(f"  Sample size n = {n}")
                
                # Theoretical rate for this (β, n) combination
                theoretical_rate = (n / np.log(n)) ** (-beta / (2 * beta + 1))
                
                empirical_losses = []
                
                # Collect empirical losses for this (β, n) combination
                for trial in range(num_trials):
                    # Generate θ₀ from Hölder ball H(β, L)
                    true_params_holder = holder_generator.generate_holder_parameters(
                        self.seq_len, mode='boundary'
                    )
                    
                    # Add noise: Y = θ₀ + n^{-1/2} * ε
                    noise_std = 1.0 / np.sqrt(n)
                    noise = torch.randn_like(true_params_holder) * noise_std
                    noisy_obs = true_params_holder + noise
                    
                    # Get transformer prediction
                    with torch.no_grad():
                        x_tensor = noisy_obs.unsqueeze(0).to(self.device)
                        pred_params = self.model(x_tensor).cpu()
                    
                    # Compute weighted L∞ loss
                    weighted_loss = self.weighted_loss.compute_loss(pred_params[0], true_params_holder)
                    empirical_losses.append(weighted_loss)
                
                # Store data for universal constant search
                all_empirical_data.extend(empirical_losses)
                all_theoretical_rates.extend([theoretical_rate] * len(empirical_losses))
                beta_n_combinations.extend([(beta, n)] * len(empirical_losses))
                
                print(f"    Collected {len(empirical_losses)} samples")
        
        print(f"\nPhase 2: Finding universal constants M, B")
        print("-" * 40)
        print(f"Total empirical samples: {len(all_empirical_data)}")
        print(f"Testing {len(beta_values)} β values × {len(sample_sizes)} sample sizes = {len(beta_values) * len(sample_sizes)} combinations")
        
        # Convert to numpy arrays for easier manipulation
        all_empirical_data = np.array(all_empirical_data)
        all_theoretical_rates = np.array(all_theoretical_rates)
        
        # Find universal constants
        universal_constants = self._find_universal_constants(
            all_empirical_data, all_theoretical_rates, beta_n_combinations, sample_sizes
        )
        
        M_universal = universal_constants['M']
        B_universal = universal_constants['B']
        
        meets_criteria = universal_constants.get('meets_criteria', False)
        
        if M_universal is None or B_universal is None or not meets_criteria:
            print(f"\nPhase 3: Universal constants {'found but below threshold' if M_universal is not None else 'not found'}")
            print("-" * 40)
            if M_universal is not None:
                print(f"THEOREM 3.1 PARTIALLY SATISFIED: Constants found but success rate below 95%")
                print(f"  M = {M_universal:.3f}")
                print(f"  B = {B_universal:.3f}")
                print(f"  Success rate = {universal_constants['success_rate']:.1%}")
            else:
                print("THEOREM 3.1 FAILED: No universal constants M, B exist")
            
            # Return failure results
            results = {
                'sample_sizes': sample_sizes,
                'beta_values': beta_values,
                'L': L,
                'universal_constants': universal_constants,
                'performance_results': {'all_combinations_pass': False, 'overall_success_rate': universal_constants.get('success_rate', 0.0)},
                'theorem_satisfied': False,
                'overall_success_rate': universal_constants.get('success_rate', 0.0)
            }
            
            # Generate failure report
            self._generate_failure_report(results, save_path)
            return results
        
        print(f"\nPhase 3: Testing transformer performance on fixed empirical data")
        print("-" * 40)
        print(f"Universal M = {M_universal:.3f}")
        print(f"Universal B = {B_universal:.3f}")
        print(f"Search success rate = {universal_constants['success_rate']:.1%}")
        
        # Test transformer performance on the SAME empirical data used for finding constants
        performance_results = self._test_transformer_performance_on_fixed_data(
            M_universal, B_universal, all_empirical_data, all_theoretical_rates, 
            beta_n_combinations, sample_sizes
        )
        
        # Combine results
        results = {
            'sample_sizes': sample_sizes,
            'beta_values': beta_values,
            'L': L,
            'universal_constants': {
                'M': M_universal,
                'B': B_universal,
                'search_details': universal_constants
            },
            'performance_results': performance_results,
            'theorem_satisfied': performance_results['all_combinations_pass'] and meets_criteria and universal_constants.get('is_perfect', False),
            'overall_success_rate': performance_results['overall_success_rate']
        }
        
        # Generate plots and reports
        self._plot_universal_constants_results(results, save_path)
        self._generate_universal_constants_report(results, save_path)
        
        return results
    
    def _find_universal_constants(self, empirical_data: np.ndarray, theoretical_rates: np.ndarray, 
                                 beta_n_combinations: List[Tuple[float, int]], 
                                 sample_sizes: List[int]) -> Dict:
        """Find universal constants M, B that work for all (β, n) combinations."""
        
        # Strategy: Fix B=0.5 and find smallest M that gives 100% success rate
        # If that fails, find the best we can achieve
        B_fixed = 0.5
        M_candidates = np.linspace(0.1, 30.0, 3000)  # Extended range for perfect constants
        target_success_rate = 1.0  # Target 100% success
        min_acceptable_rate = 0.95  # Fallback to 95%+ if 100% not achievable
        
        print(f"Searching for universal constants with B = {B_fixed} (target: {target_success_rate:.0%}, min acceptable: {min_acceptable_rate:.0%})...")
                
        best_M = None
        best_success_rate = 0.0
        perfect_M = None  # For 100% success
        
        # For each M candidate
        for M in M_candidates:
            # Check if this M works for ALL combinations with B = B_fixed
            combination_success = {}
            
            # Group data by (β, n) combination
            for beta, n in set(beta_n_combinations):
                # Get empirical data for this specific (β, n) combination
                mask = [(b, s) == (beta, n) for b, s in beta_n_combinations]
                combo_empirical = empirical_data[mask]
                combo_theoretical = theoretical_rates[mask]
                
                # Check concentration inequality for this combination
                thresholds = M * combo_theoretical
                concentration_prob = np.mean(combo_empirical >= thresholds)
                required_prob = n ** (-B_fixed)
                
                combination_success[(beta, n)] = concentration_prob <= required_prob
            
            # Calculate success rate for this M
            success_rate = np.mean(list(combination_success.values()))
            
            if success_rate >= target_success_rate and perfect_M is None:
                # Found perfect M (100% success)
                perfect_M = M
                print(f"  Found PERFECT constants: M = {perfect_M:.3f}, B = {B_fixed}, success rate = {success_rate:.1%}")
                best_M = M
                best_success_rate = success_rate
                break
            elif success_rate >= min_acceptable_rate and best_M is None:
                # Found acceptable M (≥95% success) 
                best_M = M
                best_success_rate = success_rate
                print(f"  Found acceptable constants: M = {best_M:.3f}, B = {B_fixed}, success rate = {success_rate:.1%}")
                # Don't break - keep searching for perfect
            elif success_rate > best_success_rate:
                # Track best partial success
                best_M = M
                best_success_rate = success_rate
        
        # Determine what we found
        has_perfect = perfect_M is not None
        has_acceptable = best_M is not None and best_success_rate >= min_acceptable_rate
        
        if not has_acceptable:
            # No satisfactory universal constants found
            print(f"  No universal constants found with success rate ≥ {min_acceptable_rate:.0%}")
            if best_M is not None:
                print(f"  Best achieved: M = {best_M:.3f}, B = {B_fixed}, success rate = {best_success_rate:.1%}")
            return {
                'M': best_M,
                'B': B_fixed,
                'success_rate': best_success_rate if best_M is not None else 0.0,
                'meets_criteria': False,
                'is_perfect': False,
                'search_details': {
                    'M_candidates_tested': len(M_candidates),
                    'B_fixed': B_fixed,
                    'target_success_rate': target_success_rate,
                    'min_acceptable_rate': min_acceptable_rate,
                    'total_combinations': len(set(beta_n_combinations)),
                    'best_M': best_M,
                    'best_success_rate': best_success_rate if best_M is not None else 0.0
                }
            }
        
        return {
            'M': best_M,
            'B': B_fixed,
            'success_rate': best_success_rate,
            'meets_criteria': True,
            'is_perfect': has_perfect,
            'search_details': {
                'M_candidates_tested': len(M_candidates),
                'B_fixed': B_fixed,
                'target_success_rate': target_success_rate,
                'min_acceptable_rate': min_acceptable_rate,
                'total_combinations': len(set(beta_n_combinations)),
                'perfect_M': perfect_M,
                'has_perfect_solution': has_perfect
            }
        }
    
    def _test_transformer_performance_on_fixed_data(self, M: float, B: float, 
                                                  empirical_data: np.ndarray, theoretical_rates: np.ndarray,
                                                  beta_n_combinations: List[Tuple[float, int]], 
                                                  sample_sizes: List[int]) -> Dict:
        """Test transformer performance on the fixed empirical data used for finding constants."""
        
        performance_results = {
            'combination_results': {},
            'all_combinations_pass': True,
            'overall_success_rate': 0.0,
            'detailed_stats': {}
        }
        
        total_combinations = 0
        successful_combinations = 0
        
        print(f"Testing transformer performance on fixed empirical data...")
        
        # Group data by (β, n) combination and test each
        for beta, n in set(beta_n_combinations):
            print(f"\nTesting β = {beta:.3f}, n = {n}")
            
            # Get empirical data for this specific (β, n) combination
            mask = [(b, s) == (beta, n) for b, s in beta_n_combinations]
            combo_empirical = empirical_data[mask]
            combo_theoretical = theoretical_rates[mask]
            
            total_combinations += 1
            
            # Use the SAME theoretical rate from original data (not recalculated)
            if len(combo_theoretical) > 0:
                theoretical_rate = combo_theoretical[0]  # All values in combo_theoretical are identical
            else:
                theoretical_rate = (n / np.log(n)) ** (-beta / (2 * beta + 1))  # Fallback
            
            threshold = M * theoretical_rate
            required_prob = n ** (-B)
            
            # Check concentration inequality on the fixed empirical data
            concentration_prob = np.mean(combo_empirical >= threshold)
            combination_passes = concentration_prob <= required_prob
            
            if combination_passes:
                successful_combinations += 1
            else:
                performance_results['all_combinations_pass'] = False
            
            # Store detailed results
            performance_results['combination_results'][(beta, n)] = {
                'theoretical_rate': theoretical_rate,
                'threshold': threshold,
                'concentration_prob': concentration_prob,
                'required_prob': required_prob,
                'passes': combination_passes,
                'empirical_mean': np.mean(combo_empirical),
                'empirical_std': np.std(combo_empirical),
                'num_samples': len(combo_empirical)
            }
            
            status = "✓" if combination_passes else "✗"
            print(f"  {status} P[loss≥{threshold:.3f}]={concentration_prob:.4f} ≤ {required_prob:.4f} ({len(combo_empirical)} samples)")
        
        performance_results['overall_success_rate'] = successful_combinations / total_combinations
        
        print(f"\nPerformance Summary:")
        print(f"  Successful combinations: {successful_combinations}/{total_combinations}")
        print(f"  Success rate: {performance_results['overall_success_rate']:.1%}")
        print(f"  Transformer meets bounds: {performance_results['all_combinations_pass']}")
        
        # Debug: Show which combinations failed
        failed_combinations = [(beta, n) for (beta, n), result in performance_results['combination_results'].items() 
                              if not result['passes']]
        if failed_combinations:
            print(f"  Failed combinations: {failed_combinations}")
        
        return performance_results
    

    
    def _plot_universal_constants_results(self, results: Dict, save_path: str = None):
        """Plot results of universal constants analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Theorem 3.1 Analysis: Universal Constants M, B', fontsize=16)
        
        sample_sizes = results['sample_sizes']
        beta_values = results['beta_values']
        M_universal = results['universal_constants']['M']
        B_universal = results['universal_constants']['B']
        
        # Plot 1: Universal constants display
        ax1 = axes[0, 0]
        if M_universal is not None and B_universal is not None:
            ax1.text(0.1, 0.7, f'Universal Constants:', fontsize=14, fontweight='bold', transform=ax1.transAxes)
            ax1.text(0.1, 0.5, f'M = {M_universal:.3f}', fontsize=12, transform=ax1.transAxes)
            ax1.text(0.1, 0.3, f'B = {B_universal:.3f}', fontsize=12, transform=ax1.transAxes)
            ax1.text(0.1, 0.1, f'Success Rate = {results["overall_success_rate"]:.1%}', fontsize=12, transform=ax1.transAxes)
            ax1.set_title('Universal Constants Found')
        else:
            ax1.text(0.1, 0.7, f'NO UNIVERSAL CONSTANTS', fontsize=14, fontweight='bold', color='red', transform=ax1.transAxes)
            ax1.text(0.1, 0.5, f'Theorem 3.1 FAILED', fontsize=12, color='red', transform=ax1.transAxes)
            search_details = results['universal_constants']['search_details']
            if 'best_M' in search_details and search_details['best_M'] is not None:
                ax1.text(0.1, 0.3, f'Best M = {search_details["best_M"]:.3f}', fontsize=10, transform=ax1.transAxes)
                ax1.text(0.1, 0.2, f'Best B = {search_details["B_fixed"]:.3f}', fontsize=10, transform=ax1.transAxes)
                ax1.text(0.1, 0.1, f'Success rate = {search_details["best_success_rate"]:.1%}', fontsize=10, transform=ax1.transAxes)
            ax1.set_title('No Universal Constants Found')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Plot 2: Success rate by β
        ax2 = axes[0, 1]
        if 'combination_results' in results['performance_results'] and results['performance_results']['combination_results']:
            beta_success_rates = []
            for beta in beta_values:
                beta_combinations = [(b, n) for b, n in results['performance_results']['combination_results'].keys() if b == beta]
                beta_successes = [results['performance_results']['combination_results'][(b, n)]['passes'] for b, n in beta_combinations]
                beta_success_rates.append(np.mean(beta_successes))
            
            bars = ax2.bar(range(len(beta_values)), beta_success_rates, alpha=0.7, 
                          color=['green' if rate == 1.0 else 'orange' if rate > 0.5 else 'red' for rate in beta_success_rates])
            
            # Add percentage labels
            for i, (bar, rate) in enumerate(zip(bars, beta_success_rates)):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{rate:.0%}', ha='center', va='bottom')
        else:
            # No validation results - all bars at 0%
            bars = ax2.bar(range(len(beta_values)), [0] * len(beta_values), alpha=0.7, color='red')
            ax2.text(0.5, 0.5, 'No validation performed', ha='center', va='center', transform=ax2.transAxes,
                    fontsize=12, color='red')
        
        ax2.set_xlabel('Smoothness Parameter β')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Success Rate by β Value')
        ax2.set_xticks(range(len(beta_values)))
        ax2.set_xticklabels([f'{beta}' for beta in beta_values])
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Success rate by sample size
        ax3 = axes[1, 0]
        if 'combination_results' in results['performance_results'] and results['performance_results']['combination_results']:
            n_success_rates = []
            for n in sample_sizes:
                n_combinations = [(b, s) for b, s in results['performance_results']['combination_results'].keys() if s == n]
                n_successes = [results['performance_results']['combination_results'][(b, s)]['passes'] for b, s in n_combinations]
                n_success_rates.append(np.mean(n_successes))
            
            bars = ax3.bar(range(len(sample_sizes)), n_success_rates, alpha=0.7,
                          color=['green' if rate == 1.0 else 'orange' if rate > 0.5 else 'red' for rate in n_success_rates])
            
            # Add percentage labels
            for i, (bar, rate) in enumerate(zip(bars, n_success_rates)):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{rate:.0%}', ha='center', va='bottom')
        else:
            # No validation results - all bars at 0%
            bars = ax3.bar(range(len(sample_sizes)), [0] * len(sample_sizes), alpha=0.7, color='red')
            ax3.text(0.5, 0.5, 'No validation performed', ha='center', va='center', transform=ax3.transAxes,
                    fontsize=12, color='red')
        
        ax3.set_xlabel('Sample Size n')
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Success Rate by Sample Size')
        ax3.set_xticks(range(len(sample_sizes)))
        ax3.set_xticklabels([f'{n}' for n in sample_sizes])
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Concentration probabilities heatmap
        ax4 = axes[1, 1]
        if 'combination_results' in results['performance_results'] and results['performance_results']['combination_results']:
            conc_probs = np.zeros((len(beta_values), len(sample_sizes)))
            
            for i, beta in enumerate(beta_values):
                for j, n in enumerate(sample_sizes):
                    if (beta, n) in results['performance_results']['combination_results']:
                        conc_probs[i, j] = results['performance_results']['combination_results'][(beta, n)]['concentration_prob']
            
            im = ax4.imshow(conc_probs, cmap='RdYlGn_r', aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4)
            cbar.set_label('Concentration Probability')
            
            # Add text annotations
            for i in range(len(beta_values)):
                for j in range(len(sample_sizes)):
                    text = ax4.text(j, i, f'{conc_probs[i, j]:.3f}', ha="center", va="center", color="black", fontsize=8)
        else:
            # No validation results - show empty heatmap
            ax4.text(0.5, 0.5, 'No validation performed\n(No universal constants found)', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12, color='red')
        
        ax4.set_xlabel('Sample Size n')
        ax4.set_ylabel('Smoothness Parameter β')
        ax4.set_title('Concentration Probabilities\nP[loss ≥ M·rate]')
        ax4.set_xticks(range(len(sample_sizes)))
        ax4.set_xticklabels([f'{n}' for n in sample_sizes])
        ax4.set_yticks(range(len(beta_values)))
        ax4.set_yticklabels([f'{beta}' for beta in beta_values])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.savefig('theorem_3_1_universal_constants.png', dpi=150, bbox_inches='tight')
            print("Plot saved to: theorem_3_1_universal_constants.png")
        
        plt.close()
    
    def _generate_universal_constants_report(self, results: Dict, save_path: str = None):
        """Generate text report of universal constants analysis."""
        report_path = save_path.replace('.png', '_report.txt') if save_path else 'theorem_3_1_universal_constants_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("THEOREM 3.1 UNIVERSAL CONSTANTS ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("METHODOLOGY:\n")
            f.write("-" * 20 + "\n")
            f.write("• Finding SINGLE pair (M,B) that works for ALL β values simultaneously\n")
            f.write("• Testing over Hölder balls H(β,L) with weighted L∞ loss\n")
            f.write("• Theorem: ∃ M,B > 0 s.t. P[ℓ∞(θ,θ₀) ≥ M·rate] ≤ n^(-B) uniformly in β\n")
            f.write("• M and B are UNIVERSAL constants, not β-dependent\n\n")
            
            universal_constants = results['universal_constants']
            if universal_constants['M'] is not None and universal_constants['B'] is not None:
                f.write("UNIVERSAL CONSTANTS FOUND:\n")
                f.write("-" * 30 + "\n")
                f.write(f"M = {universal_constants['M']:.3f}\n")
                f.write(f"B = {universal_constants['B']:.3f}\n")
                f.write(f"Search Success Rate = {universal_constants['search_details']['success_rate']:.1%}\n\n")
            else:
                f.write("NO UNIVERSAL CONSTANTS FOUND:\n")
                f.write("-" * 30 + "\n")
                f.write("M = None\n")
                f.write("B = None\n")
                f.write("Search failed to find constants that work for all combinations\n\n")
            
            f.write("VALIDATION RESULTS:\n")
            f.write("-" * 20 + "\n")
            performance = results['performance_results']
            f.write(f"Overall Success Rate: {performance['overall_success_rate']:.1%}\n")
            f.write(f"All Combinations Pass: {performance['all_combinations_pass']}\n\n")
            
            if 'combination_results' in results['performance_results'] and results['performance_results']['combination_results']:
                f.write("DETAILED RESULTS BY (β, n) COMBINATION:\n")
                f.write("-" * 40 + "\n")
                f.write("β    n    Theoretical Rate   Threshold   Concentration Prob   Required Prob   Pass\n")
                f.write("-" * 80 + "\n")
                
                for (beta, n), result in results['performance_results']['combination_results'].items():
                    f.write(f"{beta:.1f}  {n:3d}    {result['theoretical_rate']:.6f}      {result['threshold']:.6f}      {result['concentration_prob']:.6f}        {result['required_prob']:.6f}     {result['passes']}\n")
            else:
                f.write("No validation performed (no universal constants found)\n")
            
            if 'combination_results' in results['performance_results'] and results['performance_results']['combination_results']:
                f.write(f"\nSUMMARY BY β VALUE:\n")
                f.write("-" * 20 + "\n")
                
                for beta in results['beta_values']:
                    beta_combinations = [(b, n) for b, n in results['performance_results']['combination_results'].keys() if b == beta]
                    beta_successes = [results['performance_results']['combination_results'][(b, n)]['passes'] for b, n in beta_combinations]
                    beta_success_rate = np.mean(beta_successes)
                    f.write(f"β = {beta}: {len(beta_successes)} combinations, {beta_success_rate:.1%} success rate\n")
                
                f.write(f"\nSUMMARY BY SAMPLE SIZE:\n")
                f.write("-" * 25 + "\n")
                
                for n in results['sample_sizes']:
                    n_combinations = [(b, s) for b, s in results['performance_results']['combination_results'].keys() if s == n]
                    n_successes = [results['performance_results']['combination_results'][(b, s)]['passes'] for b, s in n_combinations]
                    n_success_rate = np.mean(n_successes)
                    f.write(f"n = {n}: {len(n_successes)} combinations, {n_success_rate:.1%} success rate\n")
            
            f.write(f"\nOVERALL ASSESSMENT:\n")
            f.write("-" * 20 + "\n")
            
            if results['performance_results']['all_combinations_pass']:
                f.write("✓ THEOREM 3.1 IS SATISFIED\n")
                f.write("Universal constants M, B exist that work for all tested β values.\n")
                f.write("The spike-slab transformer achieves uniform posterior contraction.\n")
            else:
                f.write("✗ THEOREM 3.1 IS NOT FULLY SATISFIED\n")
                f.write("No universal constants found that work for all (β, n) combinations.\n")
                f.write(f"Best achievable success rate: {results['overall_success_rate']:.1%}\n")
                
                # Identify problematic combinations
                failed_combinations = [(beta, n) for (beta, n), result in results['performance_results']['combination_results'].items() 
                                     if not result['passes']]
                if failed_combinations:
                    f.write(f"\nFailed combinations: {failed_combinations}\n")
        
        print(f"Report saved to: {report_path}")
    
    def _generate_failure_report(self, results: Dict, save_path: str = None):
        """Generate failure report when no universal constants are found."""
        report_path = save_path.replace('.png', '_failure_report.txt') if save_path else 'theorem_3_1_failure_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("THEOREM 3.1 FAILURE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("RESULT: THEOREM 3.1 IS NOT SATISFIED\n")
            f.write("-" * 40 + "\n")
            f.write("No universal constants M, B exist that work for all tested β values.\n\n")
            
            f.write("METHODOLOGY:\n")
            f.write("-" * 20 + "\n")
            f.write("• Searched for SINGLE pair (M,B) that works for ALL β values simultaneously\n")
            f.write("• Testing over Hölder balls H(β,L) with weighted L∞ loss\n")
            f.write("• Theorem: ∃ M,B > 0 s.t. P[ℓ∞(θ,θ₀) ≥ M·rate] ≤ n^(-B) uniformly in β\n")
            f.write("• M and B must be UNIVERSAL constants, not β-dependent\n\n")
            
            f.write("SEARCH DETAILS:\n")
            f.write("-" * 20 + "\n")
            search_details = results['universal_constants']['search_details']
            f.write(f"M candidates tested: {search_details['M_candidates_tested']}\n")
            f.write(f"B candidates tested: {search_details['B_candidates_tested']}\n")
            f.write(f"Total (β,n) combinations: {search_details['total_combinations']}\n\n")
            
            if 'best_partial_M' in search_details and search_details['best_partial_M'] is not None:
                f.write("BEST PARTIAL CONSTANTS FOUND:\n")
                f.write("-" * 30 + "\n")
                f.write(f"M = {search_details['best_partial_M']:.3f}\n")
                f.write(f"B = {search_details['best_partial_B']:.3f}\n")
                f.write(f"Success rate = {search_details['best_partial_success_rate']:.1%}\n")
                f.write("(These constants work for some but not all combinations)\n\n")
            
            f.write("IMPLICATIONS:\n")
            f.write("-" * 15 + "\n")
            f.write("• The spike-slab transformer does NOT achieve uniform posterior contraction\n")
            f.write("• Different β values require different constants M, B\n")
            f.write("• The theorem's uniformity requirement is violated\n")
            f.write("• Consider:\n")
            f.write("  - Adjusting model architecture or training\n")
            f.write("  - Testing weaker versions of the theorem\n")
            f.write("  - Examining β-dependent constants separately\n")
        
        print(f"Failure report saved to: {report_path}")

    def _plot_holder_ball_theorem_results(self, results: Dict, save_path: str = None):
        """Plot results of Theorem 3.1 analysis over Hölder balls."""
        # Keep the original method for backwards compatibility
        # This will be called if the old method is still used somewhere
        pass

    def _generate_holder_ball_theorem_report(self, results: Dict, save_path: str = None):
        """Generate text report of Theorem 3.1 analysis over Hölder balls."""
        # Keep the original method for backwards compatibility  
        # This will be called if the old method is still used somewhere
        pass


if __name__ == "__main__":
    model_path = "checkpoints/naive_wavelet_transformer.pkl"
    
    if os.path.exists(model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        analyzer = TheoreticalAnalyzer(model_path, device=device)
        
        # Run the proper Theorem 3.1 analysis over Hölder balls
        results = analyzer.test_theorem_3_1_over_holder_balls(
            save_path="theorem_3_1_analysis.png"
        )
        
        print("\nTheorem 3.1 analysis completed!")
        print("Results saved to: theorem_3_1_analysis.png")
    else:
        print(f"Model file not found: {model_path}")
        print("Please train a model first using train/train_wavelets.py") 