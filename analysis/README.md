# Analysis Tools for Wavelet Transformers

This directory contains tools for analyzing and visualizing trained wavelet transformer models.

## Files

- `visualize_results.py` - Main analysis class with comprehensive visualization tools
- `theoretical_analysis.py` - Theoretical analysis of Theorem 3.1 over Hölder balls
- `run_analysis.py` - Simple runner script for analyzing saved models
- `run_theorem_analysis.py` - Runner script for Theorem 3.1 analysis
- `__init__.py` - Package initialization

## Usage

### Quick Analysis
```bash
# Analyze default model
python analysis/run_analysis.py

# Analyze specific model
python analysis/run_analysis.py checkpoints/my_model.pkl

# Specify output directory
python analysis/run_analysis.py checkpoints/my_model.pkl my_analysis_results/
```

### Theoretical Analysis (Theorem 3.1)
```bash
# Test Theorem 3.1 guarantees over Hölder balls with β-interval grids
python analysis/run_theorem_analysis.py

# Specify model and output directory
python analysis/run_theorem_analysis.py checkpoints/my_model.pkl theorem_results/

# Custom sample sizes and number of trials
python analysis/run_theorem_analysis.py --sample-sizes 64 128 256 --num-trials 200
```

### Programmatic Usage
```python
from analysis import WaveletAnalyzer

# Load and analyze model
analyzer = WaveletAnalyzer('checkpoints/my_model.pkl')

# Generate full report
analyzer.generate_report('analysis_results/')

# Individual analyses
analyzer.plot_training_history()
analyzer.analyze_predictions()
analyzer.visualize_attention_patterns()
```

## Features

### Training History
- Loss curves (training and validation)
- R-squared progression
- Sparsity learning tracking

### Prediction Analysis
- Prediction vs target scatter plots
- Residual analysis
- Sparsity recovery metrics
- Coefficient distribution analysis
- Performance by wavelet level

### Attention Patterns
- Coefficient recovery visualization
- Prediction error analysis
- Sample-by-sample inspection

### Comprehensive Reports
- Automatic generation of all plots
- Summary statistics
- Detailed results saved as pickle files
- Text summary reports

### Theoretical Analysis (Theorem 3.1)
The framework provides a proper implementation of Theorem 3.1:

**Implementation:**
- Extracts L₀ from the slab prior support [-L₀, L₀]
- Generates L values systematically satisfying L₀ - 1 ≥ L > 0
- Generates β intervals [β₁, β₂] for uniform convergence testing
- Creates grids of β values within each interval  
- Tests over Hölder balls H(β, L) for each (β, L) combination
- Uses weighted ℓ∞ loss: Σⱼ 2^(j/2) maxₖ |θⱼₖ - θ'ⱼₖ|
- Finds optimal constants M, B for each configuration
- Checks uniform convergence across entire intervals and L values

**Analysis provides:**
- Posterior contraction rates: Test (n/log n)^{-β/(2β+1)} convergence
- Optimal constants M, B finding for each (β, L) combination
- Concentration probability bounds: P[deviation] ≤ n^{-B}
- Uniform convergence assessment across β intervals for each L
- Success rate evaluation per (L, β-interval) configuration
- Overall assessment across all L values and β intervals

## Model Requirements

Models must be saved as pickle files containing:
- `model`: The trained transformer model
- `wavelet_prior`: The wavelet spike-slab prior used for training
- `training_history`: Training and validation losses
- `config`: Model configuration parameters

This format is automatically created by the training script when using `save_model_pickle()`.

## Output Structure

Analysis results are saved in the specified output directory:
```
analysis_results/
├── training_history.png          # Training progress plots
├── prediction_analysis.png       # Comprehensive prediction analysis
├── attention_patterns.png        # Sample predictions and errors
├── detailed_results.pkl          # All results as pickle file
└── summary_report.txt            # Text summary

theoretical_analysis/
├── theorem_L1_interval_1_analysis.png     # Analysis for L₁, β interval 1
├── theorem_L1_interval_1_analysis_report.txt
├── theorem_L1_interval_2_analysis.png     # Analysis for L₁, β interval 2
├── theorem_L1_interval_2_analysis_report.txt
├── theorem_L1_interval_3_analysis.png     # Analysis for L₁, β interval 3
├── theorem_L1_interval_3_analysis_report.txt
├── theorem_L1_interval_4_analysis.png     # Analysis for L₁, β interval 4
├── theorem_L1_interval_4_analysis_report.txt
├── theorem_L2_interval_1_analysis.png     # Analysis for L₂, β interval 1
├── theorem_L2_interval_1_analysis_report.txt
├── ...                                    # Similar for L₂, L₃, etc.
└── theorem_results.pkl                    # Combined results for all (L, β-interval) configurations
```

## Research Questions

The analysis tools help answer:

### Basic Performance
- **Can transformers learn Bayesian structure?** Sparsity precision/recall metrics
- **What patterns did the model discover?** Attention and prediction visualizations
- **How does performance vary by resolution?** Wavelet level analysis
- **Is the model truly learning or memorizing?** Residual and correlation analysis

### Theoretical Guarantees (Theorem 3.1)
- **Do transformers achieve optimal rates?** Posterior contraction rate analysis
- **Is convergence uniform over smoothness intervals?** β-interval uniformity testing
- **Are tail bounds satisfied?** Concentration probability verification
- **Does the transformer satisfy theoretical guarantees?** Optimal constants M, B assessment
- **Which smoothness regimes work best?** Interval-by-interval success analysis
- **How does performance vary with Hölder ball radius?** L-value robustness testing
- **What is the valid range for L given slab prior support?** L₀-based constraint verification 