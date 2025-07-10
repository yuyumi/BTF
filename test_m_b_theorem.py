#!/usr/bin/env python3
"""
Test script for Theorem 3.1 universal constants M, B analysis.

This script tests whether universal constants M, B exist that work 
uniformly over a β interval [β₁, β₂], as required by Theorem 3.1.

Methodology:
1. Extract L₀ from the spike-slab prior (slab support)
2. Choose L satisfying the constraint L₀ - 1 ≥ L > 0
3. Define β interval [β₁, β₂] and sample 20 β values uniformly
4. Search for universal M, B that work for ALL (β, n) combinations
5. Validate the theorem: ∃ M, B > 0 s.t. 
   P[ℓ∞(θ,θ₀) ≥ M(n/log n)^{-β/(2β+1)}] ≤ n^{-B} uniformly in β ∈ [β₁, β₂]
"""

import os
import sys
import torch
import numpy as np
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load config
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

from analysis.theoretical_analysis import TheoreticalAnalyzer

def test_universal_constants():
    """Test the universal constants M, B finding approach over β interval."""
    
    print("=" * 70)
    print("TESTING THEOREM 3.1: UNIVERSAL CONSTANTS M, B OVER β INTERVAL")
    print("=" * 70)
    
    # Load configuration
    config = load_config()
    theorem_config = config.get('theorem_testing', {})
    
    # Check if model exists
    model_path = "checkpoints/naive_wavelet_transformer.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train a model first using:")
        print("  python train/train_wavelets.py")
        return False
    
    print(f"✓ Found model file: {model_path}")
    
    # Initialize analyzer
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        analyzer = TheoreticalAnalyzer(model_path, device=device)
        print("✓ TheoreticalAnalyzer initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return False
    
    # Extract L0 from spike-slab prior to determine valid L values
    print(f"\nExtracting Prior Parameters:")
    spike_slab_prior = analyzer.base_wavelet_prior.spike_slab_prior
    L0 = spike_slab_prior.L0  # Slab support [-L₀, L₀]
    print(f"  Spike-slab slab support L₀: {L0}")
    print(f"  Constraint: L₀ - 1 ≥ L > 0")
    print(f"  Valid L range: (0, {L0 - 1:.3f}]")
    
    # Choose L value satisfying the constraint
    if L0 - 1 <= 0:
        print(f"❌ Current constraint violated: L₀ - 1 = {L0 - 1:.3f} ≤ 0")
        print(f"   Need L₀ > 1 for theorem to be testable, but L₀ = {L0}")
        raise ValueError(f"Cannot test theorem with L₀ = {L0} ≤ 1. Please increase L₀ in spike-slab prior.")
    else:
        # Use L = L₀ - 1 (maximum valid value)
        L = L0 - 1
        print(f"  ✓ Constraint satisfied: L₀ - 1 = {L0 - 1:.3f} ≥ L = {L:.3f} > 0")
        print(f"  Selected L = {L:.3f}")
    
    # Define β interval and sample uniformly
    beta1, beta2 = theorem_config.get('beta_interval', [0.5, 2.5])
    num_betas = theorem_config.get('num_beta_samples', 20)
    beta_values = np.linspace(beta1, beta2, num_betas).tolist()
    
    # Test parameters
    print(f"\nTest Configuration:")
    sample_sizes = theorem_config.get('sample_sizes', [32, 64, 128])
    num_trials = theorem_config.get('num_trials', 15)
    
    print(f"  β interval: [{beta1}, {beta2}]")
    print(f"  Number of β values: {num_betas}")
    print(f"  β values: {[f'{b:.2f}' for b in beta_values[:5]]}...{[f'{b:.2f}' for b in beta_values[-2:]]}")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  Hölder ball radius L: {L:.3f}")
    print(f"  Trials per combination: {num_trials}")
    print(f"  Total combinations: {len(sample_sizes)} × {len(beta_values)} = {len(sample_sizes) * len(beta_values)}")
    
    # Run the analysis
    try:
        print(f"\n" + "=" * 70)
        print("RUNNING UNIVERSAL CONSTANTS ANALYSIS")
        print("=" * 70)
        print(f"Testing uniformity over β ∈ [{beta1}, {beta2}] with {num_betas} samples")
        print(f"Using L = {L:.3f} (satisfies L₀ - 1 ≥ L > 0 with L₀ = {L0})")
        print(f"Searching for constants that work for ALL {len(sample_sizes) * len(beta_values)} combinations")
        
        results = analyzer.test_theorem_3_1_over_holder_balls(
            sample_sizes=sample_sizes,
            beta_values=beta_values,
            L=L,
            num_trials=num_trials,
            save_path="test_theorem_3_1_analysis.png"
        )
        
        print(f"\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
        # Display results
        print_results_summary(results)
        
        return results['theorem_satisfied']
        
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_results_summary(results):
    """Print a summary of the results."""
    
    print(f"\nRESULTS SUMMARY:")
    print("-" * 30)
    
    # Show test configuration
    print(f"β interval tested: [{min(results['beta_values']):.2f}, {max(results['beta_values']):.2f}]")
    print(f"Number of β values: {len(results['beta_values'])}")
    print(f"Hölder ball radius L: {results['L']:.3f}")
    print(f"Sample sizes: {results['sample_sizes']}")
    print()
    
    universal_constants = results['universal_constants']
    M = universal_constants['M']
    B = universal_constants['B']
    
    if M is not None and B is not None:
        print(f"✓ UNIVERSAL CONSTANTS FOUND:")
        print(f"    M = {M:.3f}")
        print(f"    B = {B:.3f}")
        print(f"    Search success rate: {universal_constants['search_details']['success_rate']:.1%}")
        
        validation = results['validation_results']
        print(f"\n✓ VALIDATION RESULTS:")
        print(f"    Overall success rate: {validation['overall_success_rate']:.1%}")
        print(f"    All combinations pass: {validation['all_combinations_pass']}")
        
        if validation['all_combinations_pass']:
            print(f"\n🎉 THEOREM 3.1 IS SATISFIED!")
            print(f"   Universal constants M, B exist that work for all β ∈ [{min(results['beta_values']):.2f}, {max(results['beta_values']):.2f}].")
        else:
            print(f"\n⚠️  THEOREM 3.1 PARTIALLY SATISFIED")
            print(f"   Constants found but some combinations still fail.")
            
            # Show which combinations failed
            failed_combinations = []
            for (beta, n), result in validation['combination_results'].items():
                if not result['passes']:
                    failed_combinations.append((beta, n))
            
            if failed_combinations:
                print(f"   Failed combinations: {failed_combinations}")
    
    else:
        print(f"❌ NO UNIVERSAL CONSTANTS FOUND")
        print(f"   Theorem 3.1 is NOT satisfied for β ∈ [{min(results['beta_values']):.2f}, {max(results['beta_values']):.2f}]")
        
        search_details = universal_constants['search_details']
        if 'best_partial_M' in search_details and search_details['best_partial_M'] is not None:
            print(f"\n📊 BEST PARTIAL CONSTANTS:")
            print(f"    M = {search_details['best_partial_M']:.3f}")
            print(f"    B = {search_details['best_partial_B']:.3f}")
            print(f"    Partial success rate = {search_details['best_partial_success_rate']:.1%}")
        
        print(f"\n💡 IMPLICATIONS:")
        print(f"   • The transformer does NOT achieve uniform posterior contraction over [{min(results['beta_values']):.2f}, {max(results['beta_values']):.2f}]")
        print(f"   • Different β values require different constants M, B")
        print(f"   • The theorem's uniformity requirement is violated")
    
    print(f"\nFILES GENERATED:")
    print(f"  📊 test_theorem_3_1_analysis.png")
    if M is not None and B is not None:
        print(f"  📄 test_theorem_3_1_analysis_report.txt")
    else:
        print(f"  📄 test_theorem_3_1_analysis_failure_report.txt")

def main():
    """Main test function."""
    
    print("Testing Universal Constants M, B for Theorem 3.1")
    print("This tests whether the spike-slab transformer satisfies the theorem")
    print("over a β interval [β₁, β₂] with uniform sampling.")
    
    try:
        theorem_satisfied = test_universal_constants()
        
        print(f"\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        
        if theorem_satisfied:
            print("🎉 SUCCESS: Theorem 3.1 is satisfied!")
            print("   Universal constants M, B exist for the tested β interval.")
            print("   The spike-slab transformer achieves uniform posterior contraction.")
            exit_code = 0
        else:
            print("❌ FAILURE: Theorem 3.1 is not satisfied.")
            print("   No universal constants found that work for all β values in the interval.")
            print("   The uniformity requirement of the theorem is violated.")
            exit_code = 1
        
        print(f"\nFor detailed results, check the generated report files.")
        return exit_code
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Test failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 