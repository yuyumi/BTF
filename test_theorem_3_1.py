#!/usr/bin/env python3
"""
Quick test of the Theorem 3.1 implementation over Hölder balls.
This verifies that the Hölder ball generation and weighted L∞ loss work correctly.
"""

import sys
import os
import numpy as np
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis.theoretical_analysis import HolderBallGenerator, WeightedLInfLoss


def test_holder_ball_generation():
    """Test Hölder ball parameter generation."""
    print("Testing Hölder Ball Generation")
    print("-" * 30)
    
    max_resolution = 4
    beta = 2.0
    L = 1.0
    seq_len = 16
    
    generator = HolderBallGenerator(max_resolution, beta, L)
    
    # Test boundary mode
    params_boundary = generator.generate_holder_parameters(seq_len, mode='boundary')
    print(f"Boundary mode parameters shape: {params_boundary.shape}")
    
    # Check constraints
    violations = 0
    for i in range(min(seq_len, len(generator.coeff_indices))):
        j, k = generator.coeff_indices[i]
        max_allowed = L * (2 ** (-j * (beta + 0.5)))
        actual = abs(params_boundary[i, 0].item())
        
        if actual > max_allowed + 1e-10:  # Small tolerance for numerical errors
            violations += 1
            print(f"  Violation at ({j},{k}): {actual:.6f} > {max_allowed:.6f}")
    
    if violations == 0:
        print(f"✓ All {len(generator.coeff_indices)} coefficients satisfy Hölder constraints")
    else:
        print(f"✗ {violations} constraint violations found")
    
    # Print some examples
    print("\nFirst few coefficients:")
    for i in range(min(5, len(generator.coeff_indices))):
        j, k = generator.coeff_indices[i]
        max_allowed = L * (2 ** (-j * (beta + 0.5)))
        actual = params_boundary[i, 0].item()
        print(f"  ({j},{k}): {actual:.4f} (max: {max_allowed:.4f})")
    
    return violations == 0


def test_weighted_linf_loss():
    """Test weighted L∞ loss computation."""
    print("\nTesting Weighted L∞ Loss")
    print("-" * 25)
    
    max_resolution = 3
    loss_fn = WeightedLInfLoss(max_resolution)
    
    # Create simple test case
    seq_len = 8  # Should be enough for resolution 3
    theta1 = torch.zeros(seq_len, 1)
    theta2 = torch.zeros(seq_len, 1)
    
    # Set one coefficient at each resolution level
    if len(loss_fn.coeff_indices) >= 3:
        theta1[0, 0] = 1.0  # Level 0: (0,0)
        theta1[1, 0] = 0.5  # Level 1: (1,0)  
        theta1[2, 0] = 0.25 # Level 2: (2,0)
        
        theta2[0, 0] = 0.5  # Level 0: (0,0) - diff = 0.5
        theta2[1, 0] = 0.0  # Level 1: (1,0) - diff = 0.5
        theta2[2, 0] = 0.0  # Level 2: (2,0) - diff = 0.25
    
    loss = loss_fn.compute_loss(theta1, theta2)
    
    # Expected calculation:
    # Level 0: weight = 2^(0/2) = 1, max_diff = 0.5 -> contribution = 1 * 0.5 = 0.5
    # Level 1: weight = 2^(1/2) = √2 ≈ 1.414, max_diff = 0.5 -> contribution ≈ 0.707
    # Level 2: weight = 2^(2/2) = 2, max_diff = 0.25 -> contribution = 0.5
    expected = 0.5 + np.sqrt(2) * 0.5 + 2 * 0.25
    expected = 0.5 + 0.707 + 0.5  # ≈ 1.707
    
    print(f"Computed loss: {loss:.4f}")
    print(f"Expected loss: {expected:.4f}")
    print(f"Difference: {abs(loss - expected):.6f}")
    
    # Test structure info
    print(f"\nLoss function structure:")
    print(f"  Max resolution: {loss_fn.max_resolution}")
    print(f"  Coefficient indices: {loss_fn.coeff_indices[:5]}...")
    print(f"  Level ranges: {loss_fn.level_ranges}")
    
    success = abs(loss - expected) < 0.1  # Allow some tolerance
    if success:
        print("✓ Weighted L∞ loss computation appears correct")
    else:
        print("✗ Weighted L∞ loss computation may have issues")
    
    return success


def test_model_loading():
    """Test that we can load a saved model."""
    print("\nTesting Model Loading")
    print("-" * 20)
    
    model_path = 'checkpoints/naive_wavelet_transformer.pkl'
    
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        print("Please train a model first.")
        return False
    
    try:
        import pickle
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        model = saved_data['model']
        wavelet_prior = saved_data['wavelet_prior']
        
        print(f"✓ Model loaded successfully")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Wavelet prior type: {type(wavelet_prior).__name__}")
        print(f"  Max resolution: {getattr(wavelet_prior, 'max_resolution', 'unknown')}")
        print(f"  Total coefficients: {getattr(wavelet_prior, 'total_coeffs', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Theorem 3.1 Implementation")
    print("=" * 35)
    
    test_results = []
    
    # Test 1: Hölder ball generation
    test_results.append(test_holder_ball_generation())
    
    # Test 2: Weighted L∞ loss
    test_results.append(test_weighted_linf_loss())
    
    # Test 3: Model loading
    test_results.append(test_model_loading())
    
    # Summary
    print(f"\n{'='*35}")
    print("TEST SUMMARY")
    print("-" * 15)
    
    test_names = [
        "Hölder Ball Generation",
        "Weighted L∞ Loss",
        "Model Loading"
    ]
    
    for i, (name, passed) in enumerate(zip(test_names, test_results)):
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    all_passed = all(test_results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\nReady to run Theorem 3.1 analysis!")
        print("Run: python analysis/run_theorem_analysis.py")
        print("\nThe analysis will:")
        print("- Extract L₀ from the slab prior support [-L₀, L₀]")
        print("- Generate L values satisfying L₀ - 1 ≥ L > 0")
        print("- Test over β intervals [β₁, β₂] with grids within each interval")
        print("- Test all (L, β) combinations for uniform convergence")
    else:
        print("\nPlease fix failing tests before running analysis.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main()) 