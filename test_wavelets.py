#!/usr/bin/env python3
"""
Simple test runner for wavelet and spike-slab prior components.
Run this from the project root directory.
"""

import sys
import os

# Add the data directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))

def test_spike_slab_prior():
    """Test the modular spike-slab prior implementation."""
    print("=" * 60)
    print("TESTING SPIKE-SLAB PRIOR")
    print("=" * 60)
    
    try:
        from spike_slab_prior import SpikeSlabPrior, GeometricMeanWeights, NormalSlab, LaplaceSlab
        
        # Quick test
        prior = SpikeSlabPrior(n=64, K=1.0, tau=1.0, slab_distribution=NormalSlab(std=1.0))
        print(f"✓ Created spike-slab prior: {prior}")
        
        # Test coefficient sampling
        resolution_structure = {0: 1, 1: 1, 2: 2, 3: 4}
        coeffs = prior.sample_coefficients(resolution_structure)
        nonzero_count = sum(1 for v in coeffs.values() if v != 0)
        print(f"✓ Sampled {nonzero_count}/{len(coeffs)} non-zero coefficients")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_wavelet_basis():
    """Test the Haar wavelet basis implementation."""
    print("\n" + "=" * 60)
    print("TESTING WAVELET BASIS")
    print("=" * 60)
    
    try:
        from wavelet_basis import HaarWaveletBasis
        import torch
        
        # Quick test
        basis = HaarWaveletBasis(max_resolution=3)
        print(f"✓ Created wavelet basis with max resolution: {basis.max_resolution}")
        
        # Test basis functions
        x = torch.linspace(0, 1, 100)
        scaling = basis.scaling_function(x)
        wavelet = basis.wavelet_function(x)
        
        print(f"✓ Scaling function range: [{scaling.min():.1f}, {scaling.max():.1f}]")
        print(f"✓ Wavelet function range: [{wavelet.min():.1f}, {wavelet.max():.1f}]")
        
        # Test reconstruction
        coeffs = {(0, 0): 1.0, (1, 0): 0.5}
        reconstructed = basis.reconstruct_function(coeffs, x)
        print(f"✓ Function reconstruction range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_integrated_data_generator():
    """Test the integrated wavelet data generator."""
    print("\n" + "=" * 60)
    print("TESTING INTEGRATED DATA GENERATOR")
    print("=" * 60)
    
    try:
        from wavelet_data_generator import create_wavelet_prior
        import torch
        
        # Test different configurations
        configs = ['standard', 'laplace']
        
        for config in configs:
            print(f"\n--- Testing {config} configuration ---")
            
            prior = create_wavelet_prior(config, n=64, max_resolution=3)
            
            # Test data generation
            batch_size = 2
            seq_len = prior.total_coeffs
            
            x_samples = prior.sample(batch_size, seq_len)
            training_batch = prior.generate_training_batch(batch_size, seq_len)
            
            print(f"✓ Generated samples shape: {x_samples.shape}")
            print(f"✓ Training batch shapes: x={training_batch['x'].shape}, targets={training_batch['log_prob'].shape}")
            
            # Test sparsity
            sample_coeffs = training_batch['log_prob'][0]
            nonzero_count = torch.sum(torch.abs(sample_coeffs) > 1e-6).item()
            sparsity_ratio = nonzero_count / seq_len
            print(f"✓ Sparsity: {nonzero_count}/{seq_len} non-zero ({sparsity_ratio:.1%})")
            
            # Test function reconstruction
            reconstructed = prior.reconstruct_function(sample_coeffs, num_points=32)
            print(f"✓ Function reconstruction range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Bayesian Transformer Framework - Wavelet Components")
    print("=" * 60)
    
    tests = [
        test_spike_slab_prior,
        test_wavelet_basis,
        test_integrated_data_generator
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The wavelet and spike-slab implementation is working correctly.")
    else:
        print("✗ Some tests failed. Check the output above for details.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 