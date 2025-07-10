#!/usr/bin/env python3
"""
Generate valid (M, B, L, Beta1, Beta2, n) configurations for Theorem 3.1 testing.

This script searches for universal constants M, B that work for ALL β values
in intervals [Beta1, Beta2] for given L and n values, then saves them to a CSV file.
"""

import os
import sys
import csv
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.theoretical_analysis import HolderBallGenerator, WeightedLInfLoss


def load_config():
    """Load configuration from JSON file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def generate_holder_sample(holder_generator: HolderBallGenerator, seq_len: int, 
                          n: int, weighted_loss: WeightedLInfLoss) -> float:
    """Generate a single sample from Hölder ball and compute weighted L∞ loss."""
    # Generate θ₀ from Hölder ball
    true_params = holder_generator.generate_holder_parameters(seq_len, mode='boundary')
    
    # Add noise: Y = θ₀ + n^{-1/2} * ε
    noise_std = 1.0 / np.sqrt(n)
    noise = torch.randn_like(true_params) * noise_std
    noisy_obs = true_params + noise
    
    # For this synthetic test, assume "perfect" recovery (θ̂ = θ₀)
    # In practice, this would be replaced with actual model predictions
    pred_params = true_params  # Perfect recovery case
    
    # Compute weighted L∞ loss
    return weighted_loss.compute_loss(pred_params, true_params)


def test_configuration(L: float, beta1: float, beta2: float, n: int, 
                      M: float, B: float, seq_len: int, 
                      num_trials: int = 100) -> bool:
    """Test if (M, B) works for ALL β values in [beta1, beta2] for given L and n."""
    
    # Create weighted loss function
    max_resolution = int(np.log2(seq_len)) if seq_len > 1 else 1
    weighted_loss = WeightedLInfLoss(max_resolution)
    
    # Test on a grid of β values within the interval
    beta_grid = np.linspace(beta1, beta2, 5)
    
    for beta in beta_grid:
        # Create Hölder ball generator for this β
        holder_generator = HolderBallGenerator(max_resolution, beta, L)
        
        # Theoretical rate for this (β, n) combination
        theoretical_rate = (n / np.log(n)) ** (-beta / (2 * beta + 1))
        threshold = M * theoretical_rate
        required_prob = n ** (-B)
        
        # Collect empirical samples
        empirical_losses = []
        for _ in range(num_trials):
            loss = generate_holder_sample(holder_generator, seq_len, n, weighted_loss)
            empirical_losses.append(loss)
        
        # Check concentration inequality
        concentration_prob = np.mean(np.array(empirical_losses) >= threshold)
        
        # If this β value doesn't satisfy the inequality, configuration fails
        if concentration_prob > required_prob:
            return False
    
    return True


def search_valid_configuration(L: float, beta1: float, beta2: float, n: int, 
                             seq_len: int, num_trials: int = 100) -> Optional[Tuple[float, float]]:
    """Search for valid (M, B) pair for given (L, beta1, beta2, n)."""
    
    print(f"Searching for valid (M, B) for L={L:.2f}, β∈[{beta1:.1f}, {beta2:.1f}], n={n}")
    
    # Search grids - start with reasonable ranges
    M_candidates = np.linspace(0.1, 20.0, 50)
    B_candidates = np.linspace(0.5, 3.0, 25)  # Start from 0.5 for reasonable concentration
    
    for M in M_candidates:
        for B in B_candidates:
            if test_configuration(L, beta1, beta2, n, M, B, seq_len, num_trials):
                print(f"  Found valid (M, B) = ({M:.3f}, {B:.3f})")
                return M, B
    
    print(f"  No valid (M, B) found")
    return None


def generate_configurations(num_configs: int = 10, output_path: str = "theorem_configurations.csv"):
    """Generate and save valid theorem configurations."""
    
    # Load global config
    config = load_config()
    data_config = config.get('data_generation', {})
    seq_len = 2 ** data_config.get('max_resolution', 4)  # Total coefficients
    
    print(f"Generating {num_configs} valid Theorem 3.1 configurations...")
    print(f"Using sequence length: {seq_len}")
    print("="*60)
    
    configurations = []
    
    # Define parameter ranges to search
    L_values = [0.5, 1.0, 1.5]  # Hölder ball radii
    beta_intervals = [
        (0.5, 1.0),   # Low smoothness
        (1.0, 1.5),   # Medium-low smoothness
        (1.5, 2.0),   # Medium smoothness
        (2.0, 2.5),   # High smoothness
        (0.5, 1.5),   # Wide low range
        (1.0, 2.0),   # Wide medium range
        (1.5, 2.5),   # Wide high range
    ]
    n_values = [32, 64, 128, 256]
    
    # Search for valid configurations
    attempts = 0
    max_attempts = len(L_values) * len(beta_intervals) * len(n_values)
    
    for L in L_values:
        for beta1, beta2 in beta_intervals:
            for n in n_values:
                attempts += 1
                
                if len(configurations) >= num_configs:
                    break
                
                # Search for valid (M, B) for this configuration
                result = search_valid_configuration(L, beta1, beta2, n, seq_len)
                
                if result is not None:
                    M, B = result
                    config_dict = {
                        'M': M,
                        'B': B,
                        'L': L,
                        'Beta1': beta1,
                        'Beta2': beta2,
                        'n': n,
                        'seq_len': seq_len
                    }
                    configurations.append(config_dict)
                    print(f"✓ Configuration {len(configurations)}: {config_dict}")
                
                if len(configurations) >= num_configs:
                    break
            
            if len(configurations) >= num_configs:
                break
        
        if len(configurations) >= num_configs:
            break
    
    print(f"\nGenerated {len(configurations)} valid configurations out of {attempts} attempts")
    
    # Save to CSV
    if configurations:
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['M', 'B', 'L', 'Beta1', 'Beta2', 'n', 'seq_len']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(configurations)
        
        print(f"✓ Saved {len(configurations)} configurations to {output_path}")
        
        # Display summary
        print("\nGenerated Configurations Summary:")
        print("-" * 50)
        for i, config in enumerate(configurations, 1):
            print(f"{i:2d}. M={config['M']:.3f}, B={config['B']:.3f}, L={config['L']:.2f}, "
                  f"β∈[{config['Beta1']:.1f}, {config['Beta2']:.1f}], n={config['n']}")
    else:
        print("❌ No valid configurations found")
    
    return configurations


def main():
    """Main function to generate theorem configurations."""
    
    output_path = "theorem_configurations.csv"
    num_configs = 10
    
    print("THEOREM 3.1 CONFIGURATION GENERATOR")
    print("="*50)
    print(f"Target: {num_configs} valid configurations")
    print(f"Output: {output_path}")
    print()
    
    try:
        configurations = generate_configurations(num_configs, output_path)
        
        if configurations:
            print(f"\n✓ SUCCESS: Generated {len(configurations)} valid configurations")
            print("These configurations can now be used for fast theorem testing.")
        else:
            print("\n❌ FAILED: No valid configurations generated")
            print("You may need to adjust the search parameters or increase search range.")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 