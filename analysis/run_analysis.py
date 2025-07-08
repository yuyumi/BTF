#!/usr/bin/env python3
"""
Simple runner script for analyzing saved wavelet transformer models.
Usage: python analysis/run_analysis.py [model_path] [output_dir]
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.visualize_results import load_and_analyze


def main():
    parser = argparse.ArgumentParser(description='Analyze saved wavelet transformer models')
    parser.add_argument('model_path', nargs='?', 
                       default='checkpoints/naive_wavelet_transformer.pkl',
                       help='Path to saved model pickle file')
    parser.add_argument('output_dir', nargs='?', 
                       default='analysis_results',
                       help='Directory to save analysis results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick analysis with fewer samples')
    
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
        return
    
    print(f"Analyzing model: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Run analysis
    analyzer = load_and_analyze(args.model_path, args.output_dir)
    
    if analyzer:
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {args.output_dir}/")
        
        if args.quick:
            print("Quick analysis mode - for full analysis, run without --quick flag")
    else:
        print("Analysis failed!")


if __name__ == "__main__":
    main() 