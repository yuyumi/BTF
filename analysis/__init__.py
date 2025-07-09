"""
Analysis package for wavelet transformer models.

This package provides tools for analyzing and visualizing trained wavelet
transformer models, including performance metrics, sparsity learning,
theoretical guarantees, and comprehensive reporting.
"""

from .visualize_results import WaveletAnalyzer, load_and_analyze
from .theoretical_analysis import TheoreticalAnalyzer

__all__ = ['WaveletAnalyzer', 'load_and_analyze', 'TheoreticalAnalyzer'] 