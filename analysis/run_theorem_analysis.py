#!/usr/bin/env python3
"""
Runner script for Theorem 3.1 theoretical analysis.

This script tests Theorem 3.1 over Hölder balls using the proper weighted L∞ loss function.
The analysis searches for universal constants M, B that work for ALL β values simultaneously,
as required by the mathematical statement of Theorem 3.1.

The script automatically uses pre-computed configurations when available for speed, and 
generates them automatically if they don't exist. Falls back to exhaustive search as needed.

Usage:
    python analysis/run_theorem_analysis.py
    python analysis/run_theorem_analysis.py checkpoints/my_model.pkl
    python analysis/run_theorem_analysis.py checkpoints/my_model.pkl my_output_dir/
    python analysis/run_theorem_analysis.py --sample-sizes 32 64 128 --num-trials 50
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
from analysis.generate_theorem_configurations import generate_configurations
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
    
    # Check for pre-computed configurations
    config_file = "theorem_configurations.csv"
    if os.path.exists(config_file):
        print(f"✓ Found pre-computed configurations: {config_file}")
        print("Will use these configurations when available, falling back to exhaustive search otherwise")
    else:
        print(f"No pre-computed configurations found ({config_file})")
        print("Automatically generating configurations...")
        print("-" * 40)
        
        try:
            print("This may take a few minutes...")
            # Generate configurations automatically
            generated_configs = generate_configurations(num_configs=10, output_path=config_file)
            
            if generated_configs and len(generated_configs) > 0:
                print(f"✓ Successfully generated {len(generated_configs)} configurations")
                print(f"✓ Saved to {config_file} for future use")
                print("Will use these configurations for analysis")
            else:
                print("❌ Failed to generate configurations")
                print("Will fall back to exhaustive search for all configurations")
                
        except Exception as e:
            print(f"❌ Error generating configurations: {e}")
            print("Will fall back to exhaustive search for all configurations")
        
        print("-" * 40)
    
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
        print("Finding universal constants M, B that work for ALL β values simultaneously")
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
                print("Searching for universal constants M, B that work for ALL β values in this interval")
                
                # Test theorem over Hölder balls for each beta in the grid
                save_path = f"{args.output_dir}/theorem_L{L_idx+1}_interval_{i+1}_analysis.png"
                
                # First find or use universal constants
                universal_constants_result = analyzer.test_theorem_3_1_over_holder_balls(
                    sample_sizes=args.sample_sizes,
                    beta_values=beta_grid,
                    L=L,
                    num_trials=args.num_trials,
                    save_path=save_path
                )
                
                # If we found universal constants, do detailed empirical vs theoretical analysis
                if universal_constants_result['theorem_satisfied']:
                    M = universal_constants_result['universal_constants']['M']
                    B = universal_constants_result['universal_constants']['B']
                    
                    print(f"\n--- Detailed Empirical vs Theoretical Analysis ---")
                    print(f"Using universal constants: M = {M:.3f}, B = {B:.3f}")
                    
                    detailed_analysis = analyzer.analyze_empirical_vs_theoretical_rates(
                        M=M, B=B, L=L,
                        sample_sizes=args.sample_sizes,
                        beta_values=beta_grid,
                        num_trials=args.num_trials,
                        save_path=f"{args.output_dir}/detailed_L{L_idx+1}_interval_{i+1}_analysis.png"
                    )
                    
                    # Merge detailed analysis into results
                    universal_constants_result['detailed_analysis'] = detailed_analysis
                
                interval_results = universal_constants_result
                
                L_results[interval_name] = interval_results
                
                # Check if theorem holds uniformly over this interval
                # With universal constants, we check if the theorem is satisfied overall
                theorem_satisfied = interval_results.get('theorem_satisfied', False)
                overall_success_rate = interval_results.get('overall_success_rate', 0.0)
                
                print(f"Interval {interval_name}: {'✓ UNIVERSAL CONSTANTS FOUND' if theorem_satisfied else '✗ NO UNIVERSAL CONSTANTS'}")
                print(f"  Overall success rate: {overall_success_rate:.1%}")
            
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
                
                # With universal constants approach, we have a single result per interval
                theorem_satisfied = interval_data.get('theorem_satisfied', False)
                overall_success_rate = interval_data.get('overall_success_rate', 0.0)
                
                # Display universal constants if found
                if 'universal_constants' in interval_data and interval_data['universal_constants']['M'] is not None:
                    M_universal = interval_data['universal_constants']['M']
                    B_universal = interval_data['universal_constants']['B']
                    print(f"  Universal constants: M = {M_universal:.3f}, B = {B_universal:.3f}")
                else:
                    print(f"  No universal constants found")
                
                print(f"  Overall success rate: {overall_success_rate:.1%}")
                print(f"  Theorem satisfied: {'✓ YES' if theorem_satisfied else '✗ NO'}")
                
                if theorem_satisfied:
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
            print("✓ Theorem 3.1 holds with universal constants across ALL tested (L, β-interval) configurations!")
        elif overall_uniform_count > total_configurations * 0.5:
            print("~ Theorem 3.1 holds with universal constants in MAJORITY of configurations")
        elif overall_uniform_count > 0:
            print("~ Theorem 3.1 holds with universal constants in SOME configurations")
        else:
            print("✗ Theorem 3.1 does not hold with universal constants in ANY configuration")
        
        print(f"\nFor detailed analysis, check the report files in {args.output_dir}/")
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 