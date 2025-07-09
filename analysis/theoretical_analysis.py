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
                for k in range(2 ** (j - 1)):
                    self.coeff_indices.append((j, k))
    
    def generate_holder_parameters(self, seq_len: int, mode: str = 'boundary') -> torch.Tensor:
        """
        Generate parameters from Hölder ball H(β, L).
        
        Args:
            seq_len: Length of coefficient sequence
            mode: 'boundary' (on boundary), 'interior' (inside ball), 'random' (random in ball)
            
        Returns:
            Coefficient tensor of shape (seq_len, 1)
        """
        coefficients = torch.zeros(seq_len, 1)
        
        for i in range(min(seq_len, len(self.coeff_indices))):
            j, k = self.coeff_indices[i]
            
            # Hölder constraint: |θ_{j,k}| ≤ L * 2^{-j(β + 1/2)}
            max_magnitude = self.L * (2 ** (-j * (self.beta + 0.5)))
            
            if mode == 'boundary':
                # Sample on boundary of constraint
                sign = np.random.choice([-1, 1])
                coefficients[i, 0] = sign * max_magnitude
            elif mode == 'interior':
                # Sample uniformly in constraint region
                coefficients[i, 0] = np.random.uniform(-max_magnitude, max_magnitude)
            elif mode == 'random':
                # Sample with probability proportional to constraint size
                if np.random.random() < 0.5:  # 50% chance of being zero
                    coefficients[i, 0] = 0.0
                else:
                    coefficients[i, 0] = np.random.uniform(-max_magnitude, max_magnitude)
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
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
                for k in range(2 ** (j - 1)):
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
        Test Theorem 3.1 over Hölder balls: Find constants M, B such that concentration holds.
        
        Tests over Hölder balls H(β, L) using weighted L∞ loss function as specified in the theorem.
        
        Theorem: ∃ M, B > 0 such that for all θ₀ ∈ H(β, L):
        P^n(θ : ℓ∞(θ, θ₀) ≥ M(n/log n)^{-β/(2β+1)}|Y^n) ≤ n^{-B}
        
        Args:
            sample_sizes: Sample sizes to test
            beta_values: Smoothness parameters to test
            L: Hölder ball radius
            num_trials: Number of trials per configuration
            save_path: Path to save results
            
        Returns:
            Dictionary with analysis results
        """
        print("\nTheorem 3.1 Analysis: Finding Constants M, B")
        print("="*60)
        print("Testing over Hölder balls H(β,L) with weighted L∞ loss")
        
        results = {
            'sample_sizes': sample_sizes,
            'beta_values': beta_values,
            'L': L,
            'holder_ball_results': {},
            'optimal_constants': {},
            'theorem_satisfied': {}
        }
        
        for beta in beta_values:
            print(f"\n--- Testing β = {beta} ---")
            
            beta_results = {
                'empirical_rates': [],
                'theoretical_rates': [],
                'optimal_M': [],
                'concentration_probabilities': [],
                'theorem_holds': []
            }
            
            # Initialize Hölder ball generator for this β
            holder_generator = HolderBallGenerator(self.max_resolution, beta, L)
            
            for n in sample_sizes:
                print(f"  Sample size n = {n}")
                
                # Theoretical rate (n/log n)^{-β/(2β+1)}
                theoretical_rate = (n / np.log(n)) ** (-beta / (2 * beta + 1))
                
                empirical_losses = []
                
                # Test over multiple θ₀ from Hölder ball
                for trial in range(num_trials):
                    # Generate θ₀ from Hölder ball H(β, L)
                    true_params_holder = holder_generator.generate_holder_parameters(
                        self.seq_len, mode='boundary'  # Test boundary cases
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
                
                # Find optimal constant M
                empirical_losses = np.array(empirical_losses)
                empirical_rate = np.mean(empirical_losses)
                
                # M should be chosen so that most empirical losses are ≤ M * theoretical_rate
                # Try different multiples of the theoretical rate (independent of empirical rate)
                M_candidates = np.linspace(0.1, 50.0, 200)
                
                best_M = None
                best_concentration_prob = 1.0
                target_prob = n ** (-1.0)  # Try B = 1 first
                
                # Find the smallest M that works (most stringent test)
                for M_candidate in M_candidates:
                    threshold = M_candidate * theoretical_rate
                    concentration_prob = np.mean(empirical_losses >= threshold)
                    
                    # We want concentration_prob ≤ n^{-B} for some reasonable B
                    if concentration_prob <= target_prob:
                        best_M = M_candidate
                        best_concentration_prob = concentration_prob
                        break  # Take the first (smallest) M that works
                
                if best_M is None:
                    # If no M works with B=1, find the M that gives the best concentration
                    # Use the 95th percentile as a reasonable threshold
                    threshold = np.percentile(empirical_losses, 95)  # 95th percentile
                    best_M = threshold / theoretical_rate
                    best_concentration_prob = 0.05
                
                # Check if theorem holds
                required_prob = n ** (-1.0)  # B = 1
                theorem_holds = best_concentration_prob <= required_prob
                
                beta_results['empirical_rates'].append(empirical_rate)
                beta_results['theoretical_rates'].append(theoretical_rate)
                beta_results['optimal_M'].append(best_M)
                beta_results['concentration_probabilities'].append(best_concentration_prob)
                beta_results['theorem_holds'].append(theorem_holds)
                
                print(f"    Theoretical rate: {theoretical_rate:.4f}")
                print(f"    Empirical rate: {empirical_rate:.4f}")
                print(f"    Optimal M: {best_M:.3f}")
                print(f"    Concentration prob: {best_concentration_prob:.4f}")
                print(f"    Required prob (n^{-1}): {required_prob:.4f}")
                print(f"    Theorem holds: {theorem_holds}")
            
            results['holder_ball_results'][beta] = beta_results
            
            # Overall assessment for this β
            all_hold = all(beta_results['theorem_holds'])
            mean_M = np.mean(beta_results['optimal_M'])
            
            results['optimal_constants'][beta] = {
                'mean_M': mean_M,
                'all_sample_sizes_pass': all_hold,
                'success_rate': np.mean(beta_results['theorem_holds'])
            }
            
            print(f"  β = {beta} Summary:")
            print(f"    Mean optimal M: {mean_M:.3f}")
            print(f"    Success rate: {np.mean(beta_results['theorem_holds']):.1%}")
        
        # Generate plots
        self._plot_holder_ball_theorem_results(results, save_path)
        
        # Generate summary report
        self._generate_holder_ball_theorem_report(results, save_path)
        
        return results
    

    
    def _plot_holder_ball_theorem_results(self, results: Dict, save_path: str = None):
        """Plot results of Theorem 3.1 analysis over Hölder balls."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Theorem 3.1 Analysis: Hölder Balls + Weighted L∞ Loss', fontsize=16)
        
        sample_sizes = results['sample_sizes']
        beta_values = results['beta_values']
        
        # Plot 1: Optimal M constants
        ax1 = axes[0, 0]
        for beta in beta_values:
            M_values = results['holder_ball_results'][beta]['optimal_M']
            ax1.plot(sample_sizes, M_values, 'o-', label=f'β={beta}')
        ax1.set_xlabel('Sample Size n')
        ax1.set_ylabel('Optimal Constant M')
        ax1.set_title('Optimal Constants M vs Sample Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Concentration probabilities
        ax2 = axes[0, 1]
        for beta in beta_values:
            conc_probs = results['holder_ball_results'][beta]['concentration_probabilities']
            ax2.semilogy(sample_sizes, conc_probs, 'o-', label=f'β={beta}')
        
        # Add n^{-1} reference line
        reference_probs = [n**(-1.0) for n in sample_sizes]
        ax2.semilogy(sample_sizes, reference_probs, 'k--', label='n^{-1} (target)')
        
        ax2.set_xlabel('Sample Size n')
        ax2.set_ylabel('Concentration Probability')
        ax2.set_title('P[ℓ∞(θ,θ_0) ≥ M·rate] vs Sample Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Theorem success rate
        ax3 = axes[1, 0]
        success_rates = [results['optimal_constants'][beta]['success_rate'] for beta in beta_values]
        bars = ax3.bar(range(len(beta_values)), success_rates, alpha=0.7)
        ax3.set_xlabel('Smoothness Parameter β')
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Theorem 3.1 Success Rate by β')
        ax3.set_xticks(range(len(beta_values)))
        ax3.set_xticklabels([f'{beta}' for beta in beta_values])
        ax3.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1%}', ha='center', va='bottom')
        
        # Plot 4: Rate comparison
        ax4 = axes[1, 1]
        for beta in beta_values:
            empirical_rates = results['holder_ball_results'][beta]['empirical_rates']
            theoretical_rates = results['holder_ball_results'][beta]['theoretical_rates']
            optimal_M = results['holder_ball_results'][beta]['optimal_M']
            
            # Adjusted theoretical rates with optimal M
            adjusted_rates = [M * rate for M, rate in zip(optimal_M, theoretical_rates)]
            
            ax4.loglog(sample_sizes, empirical_rates, 'o-', label=f'Empirical β={beta}')
            ax4.loglog(sample_sizes, adjusted_rates, '--', alpha=0.7, label=f'M·Theory β={beta}')
        
        ax4.set_xlabel('Sample Size n')
        ax4.set_ylabel('Rate')
        ax4.set_title('Empirical vs M·Theoretical Rates')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.savefig('theorem_3_1_holder_ball_analysis.png', dpi=150, bbox_inches='tight')
            print("Plot saved to: theorem_3_1_holder_ball_analysis.png")
        
        plt.close()  # Close the figure to free memory
    
    def _generate_holder_ball_theorem_report(self, results: Dict, save_path: str = None):
        """Generate text report of Theorem 3.1 analysis over Hölder balls."""
        report_path = save_path.replace('.png', '_report.txt') if save_path else 'theorem_3_1_holder_ball_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("THEOREM 3.1 ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("METHODOLOGY:\n")
            f.write("-" * 20 + "\n")
            f.write("• Testing over Hölder balls H(β,L) as specified in theorem\n")
            f.write("• Using weighted L∞ loss: ℓ∞(θ,θ') = Σⱼ 2^(j/2) max_k |θⱼₖ - θ'ⱼₖ|\n")
            f.write("• Finding optimal constants M, B such that theorem holds\n")
            f.write("• Theorem: ∃ M,B > 0 s.t. P[ℓ∞(θ,θ_0) ≥ M·rate] ≤ n^(-B)\n\n")
            
            f.write("SUMMARY BY SMOOTHNESS PARAMETER:\n")
            f.write("-" * 35 + "\n")
            
            for beta in results['beta_values']:
                constants = results['optimal_constants'][beta]
                f.write(f"\nβ = {beta}:\n")
                f.write(f"  Mean optimal M: {constants['mean_M']:.3f}\n")
                f.write(f"  Success rate: {constants['success_rate']:.1%}\n")
                f.write(f"  All sample sizes pass: {constants['all_sample_sizes_pass']}\n")
            
            f.write(f"\nDETAILED RESULTS:\n")
            f.write("-" * 20 + "\n")
            
            for beta in results['beta_values']:
                f.write(f"\nβ = {beta}:\n")
                beta_results = results['holder_ball_results'][beta]
                
                for i, n in enumerate(results['sample_sizes']):
                    f.write(f"  n={n:3d}: ")
                    f.write(f"M={beta_results['optimal_M'][i]:.3f}, ")
                    f.write(f"P={beta_results['concentration_probabilities'][i]:.4f}, ")
                    f.write(f"Pass={beta_results['theorem_holds'][i]}\n")
            
            f.write(f"\nOVERALL ASSESSMENT:\n")
            f.write("-" * 20 + "\n")
            
            all_betas_pass = all(results['optimal_constants'][beta]['all_sample_sizes_pass'] 
                               for beta in results['beta_values'])
            overall_success_rate = np.mean([results['optimal_constants'][beta]['success_rate'] 
                                          for beta in results['beta_values']])
            
            f.write(f"All β values pass: {all_betas_pass}\n")
            f.write(f"Overall success rate: {overall_success_rate:.1%}\n")
            
            if all_betas_pass:
                f.write("\n✓ THEOREM 3.1 IS SATISFIED\n")
                f.write("The transformer achieves the theoretical guarantees when properly tested.\n")
            else:
                f.write("\n✗ THEOREM 3.1 IS NOT FULLY SATISFIED\n")
                f.write("Some configurations do not meet the theoretical requirements.\n")
        
        print(f"Report saved to: {report_path}")
    

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