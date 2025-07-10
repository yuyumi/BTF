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
    """Test Hölder ball parameter generation with improved sampling."""
    print("Testing Hölder Ball Generation")
    print("-" * 30)
    
    max_resolution = 4
    beta = 2.0
    L = 1.0
    seq_len = 16
    
    generator = HolderBallGenerator(max_resolution, beta, L)
    
    all_tests_passed = True
    
    # Test 1: Boundary mode
    print("\n1. Testing boundary mode...")
    params_boundary = generator.generate_holder_parameters(seq_len, mode='boundary')
    
    # Check constraints are satisfied
    violations = 0
    boundary_count = 0
    for i in range(min(seq_len, len(generator.coeff_indices))):
        j, k = generator.coeff_indices[i]
        max_allowed = L * (2 ** (-j * (beta + 0.5)))
        actual = abs(params_boundary[i, 0].item())
        
        if actual > max_allowed + 1e-10:
            violations += 1
            print(f"  Violation at ({j},{k}): {actual:.6f} > {max_allowed:.6f}")
        
        # Check if coefficient is at boundary (within tolerance)
        if abs(actual - max_allowed) < 1e-10:
            boundary_count += 1
    
    if violations == 0:
        print(f"  ✓ All {len(generator.coeff_indices)} coefficients satisfy Hölder constraints")
    else:
        print(f"  ✗ {violations} constraint violations found")
        all_tests_passed = False
    
    if boundary_count >= 1:
        print(f"  ✓ {boundary_count} coefficient(s) at boundary (expected ≥ 1)")
    else:
        print(f"  ✗ No coefficients at boundary (expected ≥ 1)")
        all_tests_passed = False
    
    # Test 2: Interior mode
    print("\n2. Testing interior mode...")
    params_interior = generator.generate_holder_parameters(seq_len, mode='interior')
    
    violations = 0
    boundary_count = 0
    for i in range(min(seq_len, len(generator.coeff_indices))):
        j, k = generator.coeff_indices[i]
        max_allowed = L * (2 ** (-j * (beta + 0.5)))
        actual = abs(params_interior[i, 0].item())
        
        if actual > max_allowed + 1e-10:
            violations += 1
        
        # Check if coefficient is at boundary
        if abs(actual - max_allowed) < 1e-10:
            boundary_count += 1
    
    if violations == 0:
        print(f"  ✓ All coefficients satisfy Hölder constraints")
    else:
        print(f"  ✗ {violations} constraint violations found")
        all_tests_passed = False
    
    if boundary_count == 0:
        print(f"  ✓ No coefficients at boundary (expected for interior mode)")
    else:
        print(f"  ✗ {boundary_count} coefficient(s) at boundary (expected 0 for interior)")
        all_tests_passed = False
    
    # Test 3: Uniform mode
    print("\n3. Testing uniform mode...")
    params_uniform = generator.generate_holder_parameters(seq_len, mode='uniform')
    
    violations = 0
    for i in range(min(seq_len, len(generator.coeff_indices))):
        j, k = generator.coeff_indices[i]
        max_allowed = L * (2 ** (-j * (beta + 0.5)))
        actual = abs(params_uniform[i, 0].item())
        
        if actual > max_allowed + 1e-10:
            violations += 1
    
    if violations == 0:
        print(f"  ✓ All coefficients satisfy Hölder constraints")
    else:
        print(f"  ✗ {violations} constraint violations found")
        all_tests_passed = False
    
    # Print examples from boundary mode
    print("\nBoundary mode examples:")
    for i in range(min(5, len(generator.coeff_indices))):
        j, k = generator.coeff_indices[i]
        max_allowed = L * (2 ** (-j * (beta + 0.5)))
        actual = params_boundary[i, 0].item()
        at_boundary = abs(abs(actual) - max_allowed) < 1e-10
        print(f"  ({j},{k}): {actual:.6f} (max: {max_allowed:.6f}) {'[BOUNDARY]' if at_boundary else ''}")
    
    return all_tests_passed


def test_boundary_sampling_statistics():
    """Test statistical properties of boundary sampling."""
    print("\nTesting Boundary Sampling Statistics")
    print("-" * 35)
    
    max_resolution = 3
    beta = 1.5
    L = 1.0
    num_samples = 1000
    
    generator = HolderBallGenerator(max_resolution, beta, L)
    
    # Use the full coefficient count for a proper test
    seq_len = len(generator.coeff_indices)
    
    # Count how often each coefficient is at boundary
    boundary_counts = np.zeros(len(generator.coeff_indices))
    
    for _ in range(num_samples):
        params = generator.generate_holder_parameters(seq_len, mode='boundary')
        
        for i in range(len(generator.coeff_indices)):
            j, k = generator.coeff_indices[i]
            max_allowed = L * (2 ** (-j * (beta + 0.5)))
            actual = abs(params[i, 0].item())
            
            if abs(actual - max_allowed) < 1e-10:
                boundary_counts[i] += 1
    
    # Check that boundary sampling is approximately uniform
    expected_prob = 1.0 / len(generator.coeff_indices)
    actual_probs = boundary_counts / num_samples
    
    print(f"Expected probability per coefficient: {expected_prob:.3f}")
    print("Actual probabilities:")
    
    uniform_test_passed = True
    for i in range(len(generator.coeff_indices)):
        j, k = generator.coeff_indices[i]
        prob = actual_probs[i]
        print(f"  ({j},{k}): {prob:.3f}")
        
        # Allow 50% deviation from expected for reasonable tolerance
        if abs(prob - expected_prob) > 0.5 * expected_prob:
            uniform_test_passed = False
    
    if uniform_test_passed:
        print("✓ Boundary sampling appears approximately uniform")
    else:
        print("✗ Boundary sampling distribution may be biased")
    
    # Check that exactly one coefficient is at boundary in each sample
    exactly_one_count = 0
    for _ in range(100):  # Smaller sample for this test
        params = generator.generate_holder_parameters(seq_len, mode='boundary')
        
        boundary_count = 0
        for i in range(len(generator.coeff_indices)):
            j, k = generator.coeff_indices[i]
            max_allowed = L * (2 ** (-j * (beta + 0.5)))
            actual = abs(params[i, 0].item())
            
            if abs(actual - max_allowed) < 1e-10:
                boundary_count += 1
        
        if boundary_count == 1:
            exactly_one_count += 1
    
    exactly_one_prob = exactly_one_count / 100
    print(f"\nProbability of exactly one coefficient at boundary: {exactly_one_prob:.2f}")
    
    if exactly_one_prob > 0.95:
        print("✓ Boundary sampling correctly places exactly one coefficient at boundary")
        return True
    else:
        print("✗ Boundary sampling doesn't consistently place exactly one coefficient at boundary")
        return False


def test_weighted_linf_loss():
    """Test weighted L∞ loss computation."""
    print("\nTesting Weighted L∞ Loss")
    print("-" * 25)
    
    max_resolution = 3
    loss_fn = WeightedLInfLoss(max_resolution)
    
    # Create test case with enough coefficients for all levels
    seq_len = len(loss_fn.coeff_indices)
    theta1 = torch.zeros(seq_len, 1)
    theta2 = torch.zeros(seq_len, 1)
    
    # Set coefficients at the start of each level range based on level_ranges
    # Level ranges: {0: (0, 1), 1: (1, 3), 2: (3, 7), 3: (7, 15)}
    level_ranges = loss_fn.level_ranges
    
    # Set one coefficient per level for testing
    if 0 in level_ranges:
        start_idx = level_ranges[0][0]  # Index 0
        theta1[start_idx, 0] = 1.0
        theta2[start_idx, 0] = 0.5  # diff = 0.5
    
    if 1 in level_ranges:
        start_idx = level_ranges[1][0]  # Index 1
        theta1[start_idx, 0] = 0.5
        theta2[start_idx, 0] = 0.0  # diff = 0.5
    
    if 2 in level_ranges:
        start_idx = level_ranges[2][0]  # Index 3
        theta1[start_idx, 0] = 0.25
        theta2[start_idx, 0] = 0.0  # diff = 0.25
    
    loss = loss_fn.compute_loss(theta1, theta2)
    
    # Expected calculation based on actual level structure:
    # Level 0: weight = 2^(0/2) = 1, max_diff = 0.5 -> contribution = 1 * 0.5 = 0.5
    # Level 1: weight = 2^(1/2) = √2 ≈ 1.414, max_diff = 0.5 -> contribution ≈ 0.707
    # Level 2: weight = 2^(2/2) = 2, max_diff = 0.25 -> contribution = 0.5
    expected = 1.0 * 0.5 + np.sqrt(2) * 0.5 + 2.0 * 0.25
    
    print(f"Computed loss: {loss:.4f}")
    print(f"Expected loss: {expected:.4f}")
    print(f"Difference: {abs(loss - expected):.6f}")
    
    # Debug info
    print(f"\nDetailed calculation:")
    print(f"  Level 0: weight={1.0:.3f}, max_diff={0.5:.3f}, contrib={1.0*0.5:.3f}")
    print(f"  Level 1: weight={np.sqrt(2):.3f}, max_diff={0.5:.3f}, contrib={np.sqrt(2)*0.5:.3f}")
    print(f"  Level 2: weight={2.0:.3f}, max_diff={0.25:.3f}, contrib={2.0*0.25:.3f}")
    
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
    
    # Test 2: Boundary sampling statistics
    test_results.append(test_boundary_sampling_statistics())
    
    # Test 3: Weighted L∞ loss
    test_results.append(test_weighted_linf_loss())
    
    # Test 4: Model loading
    test_results.append(test_model_loading())
    
    # Summary
    print(f"\n{'='*35}")
    print("TEST SUMMARY")
    print("-" * 15)
    
    test_names = [
        "Hölder Ball Generation",
        "Boundary Sampling Statistics",
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
        print("- Use improved Hölder ball sampling with proper boundary sampling")
    else:
        print("\nPlease fix failing tests before running analysis.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main()) 