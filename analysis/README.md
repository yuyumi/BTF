# Analysis Tools for Wavelet Transformers

This directory contains tools for analyzing and visualizing trained wavelet transformer models.

## Files

- `visualize_results.py` - Main analysis class with comprehensive visualization tools
- `run_analysis.py` - Simple runner script for analyzing saved models
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
```

## Research Questions

The analysis tools help answer:
- **Can transformers learn Bayesian structure?** Sparsity precision/recall metrics
- **What patterns did the model discover?** Attention and prediction visualizations
- **How does performance vary by resolution?** Wavelet level analysis
- **Is the model truly learning or memorizing?** Residual and correlation analysis 