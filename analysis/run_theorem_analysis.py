#!/usr/bin/env python3
"""
Runner script for Theorem 3.1 theoretical analysis.

This script tests Theorem 3.1 over Hölder balls using the proper weighted L∞ loss function.

Usage:
    python analysis/run_theorem_analysis.py
    python analysis/run_theorem_analysis.py checkpoints/my_model.pkl
    python analysis/run_theorem_analysis.py checkpoints/my_model.pkl my_output_dir/
    python analysis/run_theorem_analysis.py --beta-values 1.0 2.0 3.0
"""

import os
import sys
import argparse
from pathlib import Path
import pickle
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.theoretical_analysis import TheoreticalAnalyzer
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Analyze Theorem 3.1 theoretical guarantees')
    parser.add_argument('model_path', nargs='?', 
                       default='checkpoints/naive_wavelet_transformer.pkl',
                       help='Path to saved model pickle file')
    parser.add_argument('output_dir', nargs='?', 
                       default='theoretical_analysis',
                       help='Directory to save analysis results')

    parser.add_argument('--sample-sizes', nargs='+', type=int, 
                       default=[32, 64, 128, 256, 512],
                       help='Sample sizes to test')
    parser.add_argument('--num-trials', type=int, default=100,
                       help='Number of trials per configuration')

    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        print("\nAvailable model files:")
        
        # Check common locations
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            pkl_files = list(checkpoints_dir.glob("*.pkl"))
            if pkl_files:
                for pkl_file in pkl_files:
                    print(f"  {pkl_file}")
            else:
                print("  No .pkl files found in checkpoints/")
        else:
            print("  checkpoints/ directory not found")
        
        print("\nPlease train a model first using:")
        print("  python train/train_wavelets.py")
        return 1
    
    print("="*80)
    print("THEOREM 3.1 THEORETICAL ANALYSIS")
    print("Testing posterior contraction rates over Hölder balls")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Sample sizes: {args.sample_sizes}")
    print(f"Trials per config: {args.num_trials}")
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    try:
        analyzer = TheoreticalAnalyzer(args.model_path)
        
        # Extract L0 from the loaded model's spike-slab prior
        L0 = analyzer.base_wavelet_prior.spike_slab_prior.L0
        print(f"Slab prior support: [-{L0}, {L0}]")
        
        # Generate L values systematically based on L0
        # Constraint: L₀ - 1 ≥ L > 0
        max_L = max(0.1, L0 - 1.0)  # Ensure we have at least some positive range
        if max_L <= 0:
            print(f"WARNING: L0={L0} gives max_L={max_L} ≤ 0. Using L0=2.0 instead.")
            L0 = 2.0  # Use larger L0 to get valid range
            max_L = L0 - 1.0
        
        L_values = np.linspace(0.1, max_L, 3)  # 3 L values in valid range
        print(f"Hölder ball radius L values: {L_values}")
        
        print(f"\n{'='*60}")
        print("RUNNING THEOREM 3.1 ANALYSIS")
        print("Testing over Hölder balls H(β,L) with β grids in intervals and L grid")
        print(f"{'='*60}")
        
        # Generate beta intervals [β₁, β₂] for uniform convergence testing
        beta_intervals = [
            (0.5, 1.5),   # Low smoothness interval
            (1.0, 2.0),   # Medium smoothness interval  
            (1.5, 2.5),   # High smoothness interval
            (2.0, 3.0),   # Very high smoothness interval
        ]
        
        print("Testing over β intervals:")
        for i, (beta1, beta2) in enumerate(beta_intervals):
            print(f"  Interval {i+1}: [{beta1}, {beta2}]")
        
        # For each interval and L value, create a grid of beta values and test theorem
        all_results = {}
        
        for L_idx, L in enumerate(L_values):
            print(f"\n{'='*40}")
            print(f"TESTING L = {L:.2f}")
            print(f"{'='*40}")
            
            L_results = {}
            
            for i, (beta1, beta2) in enumerate(beta_intervals):
                interval_name = f"[{beta1}, {beta2}]"
                print(f"\n--- Testing interval {interval_name} with L={L:.2f} ---")
                
                # Create beta grid within the interval
                beta_grid = np.linspace(beta1, beta2, 5)  # 5 points per interval
                print(f"Beta grid: {beta_grid}")
                
                # Test theorem over Hölder balls for each beta in the grid
                save_path = f"{args.output_dir}/theorem_L{L_idx+1}_interval_{i+1}_analysis.png"
                
                interval_results = analyzer.test_theorem_3_1_over_holder_balls(
                    sample_sizes=args.sample_sizes,
                    beta_values=beta_grid,
                    L=L,
                    num_trials=args.num_trials,
                    save_path=save_path
                )
                
                L_results[interval_name] = interval_results
                
                # Check if theorem holds uniformly over this interval
                success_rates = [interval_results['optimal_constants'][beta]['success_rate'] 
                               for beta in beta_grid]
                uniform_success = all(rate > 0.8 for rate in success_rates)
                
                print(f"Interval {interval_name}: {'✓ UNIFORM SUCCESS' if uniform_success else '✗ NON-UNIFORM'}")
            
            all_results[f"L={L:.2f}"] = L_results
        
        # Combine results
        results = {
            'L0': L0,
            'L_values': L_values,
            'beta_intervals': beta_intervals,
            'L_results': all_results
        }
        
        # Save detailed results
        results_path = f"{args.output_dir}/theorem_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED!")
        print("="*80)
        print(f"Results saved to: {args.output_dir}/")
        print("\nAnalysis Files:")
        for L_idx in range(len(L_values)):
            for i in range(len(beta_intervals)):
                print(f"  - theorem_L{L_idx+1}_interval_{i+1}_analysis.png")
                print(f"  - theorem_L{L_idx+1}_interval_{i+1}_analysis_report.txt")
        print("  - theorem_results.pkl")
        
        # Print summary
        print("\nANALYSIS SUMMARY:")
        print("-" * 40)
        
        overall_uniform_count = 0
        total_configurations = len(L_values) * len(beta_intervals)
        
        for L_idx, L in enumerate(L_values):
            L_key = f"L={L:.2f}"
            L_data = all_results[L_key]
            
            print(f"\n{'='*30}")
            print(f"L = {L:.2f} (max allowed: {max_L:.2f})")
            print(f"{'='*30}")
            
            L_uniform_intervals = 0
            
            for i, (beta1, beta2) in enumerate(beta_intervals):
                interval_name = f"[{beta1}, {beta2}]"
                interval_data = L_data[interval_name]
                
                print(f"\nInterval {i+1}: {interval_name}")
                
                # Get beta grid for this interval
                beta_grid = np.linspace(beta1, beta2, 5)
                
                interval_success_rates = []
                for beta in beta_grid:
                    success_rate = interval_data['optimal_constants'][beta]['success_rate']
                    mean_M = interval_data['optimal_constants'][beta]['mean_M']
                    print(f"  β = {beta:.2f}: Success rate = {success_rate:.1%}, Mean M = {mean_M:.3f}")
                    interval_success_rates.append(success_rate)
                
                # Check if uniform across interval
                uniform_success = all(rate > 0.8 for rate in interval_success_rates)
                avg_rate = np.mean(interval_success_rates)
                
                print(f"  Interval average: {avg_rate:.1%}")
                print(f"  Uniform success: {'✓ YES' if uniform_success else '✗ NO'}")
                
                if uniform_success:
                    L_uniform_intervals += 1
                    overall_uniform_count += 1
            
            print(f"\nL = {L:.2f} Summary: {L_uniform_intervals}/{len(beta_intervals)} intervals uniform")
        
        print(f"\n{'='*50}")
        print(f"OVERALL ASSESSMENT:")
        print(f"{'='*50}")
        print(f"Total configurations: {total_configurations}")
        print(f"Uniform configurations: {overall_uniform_count}")
        print(f"Success rate: {overall_uniform_count/total_configurations:.1%}")
        
        if overall_uniform_count == total_configurations:
            print("✓ Theorem 3.1 holds uniformly across ALL tested (L, β-interval) configurations!")
        elif overall_uniform_count > total_configurations * 0.5:
            print("~ Theorem 3.1 holds uniformly in MAJORITY of configurations")
        elif overall_uniform_count > 0:
            print("~ Theorem 3.1 holds uniformly in SOME configurations")
        else:
            print("✗ Theorem 3.1 does not hold uniformly in ANY configuration")
        
        print(f"\nFor detailed analysis, check the report files in {args.output_dir}/")
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 