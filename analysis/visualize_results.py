import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.wavelet_data_generator import create_wavelet_prior, WaveletSpikeSlabPrior
from train.train_wavelets import WaveletDataset, WaveletTrainer
from models.transformer import BayesianTransformer

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
        
sns.set_palette("husl")


class WaveletAnalyzer:
    """Comprehensive analysis and visualization for wavelet transformer models."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize analyzer with saved model.
        
        Args:
            model_path: Path to saved pickle file containing model and metadata
            device: Device to run analysis on
        """
        self.device = device
        self.model_path = model_path
        
        # Load model and metadata
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            self.saved_data = pickle.load(f)
        
        # Extract components
        self.model = self.saved_data['model']
        self.model.to(device)
        self.model.eval()
        
        self.wavelet_prior = self.saved_data['wavelet_prior']
        self.training_history = self.saved_data.get('training_history', {})
        self.config = self.saved_data.get('config', {})
        
        print(f"Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Wavelet config: {self.wavelet_prior.get_info()}")
    
    def plot_training_history(self, save_path: str = None, figsize: Tuple[int, int] = (15, 5)):
        """Plot training and validation loss curves."""
        if not self.training_history:
            print("No training history available")
            return
        
        train_losses = self.training_history.get('train_losses', [])
        val_losses = self.training_history.get('val_losses', [])
        
        if not train_losses or not val_losses:
            print("No loss data available")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        axes[0].plot(epochs, [x['loss'] for x in train_losses], label='Train Loss', linewidth=2)
        axes[0].plot(epochs, [x['loss'] for x in val_losses], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # R-squared progression
        if 'r_squared' in train_losses[0]:
            axes[1].plot(epochs, [x['r_squared'] for x in train_losses], label='Train R²', linewidth=2)
            axes[1].plot(epochs, [x['r_squared'] for x in val_losses], label='Val R²', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('R-squared')
            axes[1].set_title('Model Performance')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Sparsity tracking
        if 'pred_sparsity' in train_losses[0]:
            axes[2].plot(epochs, [x['true_sparsity'] for x in val_losses], 
                        label='True Sparsity', linewidth=2, linestyle='--')
            axes[2].plot(epochs, [x['pred_sparsity'] for x in val_losses], 
                        label='Predicted Sparsity', linewidth=2)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Sparsity Ratio')
            axes[2].set_title('Sparsity Learning')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_predictions(self, num_samples: int = 100, save_path: str = None):
        """Analyze model predictions vs true coefficients."""
        
        # Generate test data
        seq_len = self.wavelet_prior.total_coeffs
        test_dataset = WaveletDataset(self.wavelet_prior, num_samples, seq_len)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Collect predictions
        all_predictions = []
        all_targets = []
        all_inputs = []
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(self.device)
                targets = batch['log_prob'].to(self.device)
                predictions = self.model(x)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_inputs.append(x.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        inputs = np.concatenate(all_inputs, axis=0)
        
        # Flatten for analysis
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        input_flat = inputs.flatten()
        
        # Create comprehensive analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Prediction vs Target scatter
        axes[0,0].scatter(target_flat, pred_flat, alpha=0.6, s=1)
        axes[0,0].plot([target_flat.min(), target_flat.max()], 
                       [target_flat.min(), target_flat.max()], 'r--', linewidth=2)
        axes[0,0].set_xlabel('True Coefficients')
        axes[0,0].set_ylabel('Predicted Coefficients')
        axes[0,0].set_title('Prediction vs Target')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add correlation info
        corr = np.corrcoef(target_flat, pred_flat)[0, 1]
        axes[0,0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=axes[0,0].transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Residuals plot
        residuals = pred_flat - target_flat
        axes[0,1].scatter(target_flat, residuals, alpha=0.6, s=1)
        axes[0,1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0,1].set_xlabel('True Coefficients')
        axes[0,1].set_ylabel('Residuals (Pred - True)')
        axes[0,1].set_title('Residual Analysis')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Sparsity analysis
        true_zeros = np.abs(target_flat) < 1e-6
        pred_near_zeros = np.abs(pred_flat) < 0.1
        
        sparsity_data = [
            ['True Zeros', np.sum(true_zeros)],
            ['Pred Near-Zero', np.sum(pred_near_zeros)],
            ['Correct Zeros', np.sum(true_zeros & pred_near_zeros)],
            ['False Zeros', np.sum(pred_near_zeros & ~true_zeros)]
        ]
        
        categories = [x[0] for x in sparsity_data]
        values = [x[1] for x in sparsity_data]
        
        bars = axes[0,2].bar(categories, values, color=['blue', 'orange', 'green', 'red'])
        axes[0,2].set_ylabel('Count')
        axes[0,2].set_title('Sparsity Recovery Analysis')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        total = len(target_flat)
        for bar, value in zip(bars, values):
            axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                          f'{value}\n({100*value/total:.1f}%)', 
                          ha='center', va='bottom', fontsize=10)
        
        # 4. Coefficient magnitude distribution
        axes[1,0].hist(target_flat, bins=50, alpha=0.7, label='True', density=True)
        axes[1,0].hist(pred_flat, bins=50, alpha=0.7, label='Predicted', density=True)
        axes[1,0].set_xlabel('Coefficient Value')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Coefficient Distribution')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_yscale('log')
        
        # 5. Noise vs Signal recovery
        noise_level = input_flat - target_flat  # Noise in observations
        signal_recovery = pred_flat - target_flat  # Error in predictions
        
        axes[1,1].scatter(noise_level, signal_recovery, alpha=0.6, s=1)
        axes[1,1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1,1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1,1].set_xlabel('Input Noise (Observation - True)')
        axes[1,1].set_ylabel('Prediction Error (Pred - True)')
        axes[1,1].set_title('Noise vs Prediction Error')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Wavelet level analysis
        info = self.wavelet_prior.get_info()
        level_performance = {}
        
        idx = 0
        for level, count in info['resolution_structure'].items():
            level_targets = target_flat[idx:idx+count]
            level_preds = pred_flat[idx:idx+count]
            
            level_mse = np.mean((level_preds - level_targets) ** 2)
            level_corr = np.corrcoef(level_targets, level_preds)[0, 1] if len(level_targets) > 1 else 0
            level_sparsity = np.mean(np.abs(level_targets) < 1e-6)
            
            level_performance[level] = {
                'mse': level_mse,
                'correlation': level_corr if not np.isnan(level_corr) else 0,
                'sparsity': level_sparsity
            }
            idx += count
        
        levels = list(level_performance.keys())
        level_mse = [level_performance[l]['mse'] for l in levels]
        level_corr = [level_performance[l]['correlation'] for l in levels]
        
        x = np.arange(len(levels))
        width = 0.35
        
        ax2 = axes[1,2]
        ax2.bar(x - width/2, level_mse, width, label='MSE', alpha=0.7)
        ax2.set_xlabel('Wavelet Level')
        ax2.set_ylabel('MSE')
        ax2.set_title('Performance by Wavelet Level')
        ax2.set_xticks(x)
        ax2.set_xticklabels(levels)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add correlation on secondary axis
        ax2_twin = ax2.twinx()
        ax2_twin.bar(x + width/2, level_corr, width, label='Correlation', alpha=0.7, color='orange')
        ax2_twin.set_ylabel('Correlation')
        ax2_twin.legend(loc='upper right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("PREDICTION ANALYSIS SUMMARY")
        print("="*60)
        print(f"Overall Correlation: {corr:.3f}")
        print(f"Overall MSE: {np.mean(residuals**2):.4f}")
        print(f"Overall MAE: {np.mean(np.abs(residuals)):.4f}")
        print(f"True Sparsity: {np.mean(true_zeros):.1%}")
        print(f"Predicted Sparsity: {np.mean(pred_near_zeros):.1%}")
        
        if np.sum(true_zeros) > 0:
            precision = np.sum(true_zeros & pred_near_zeros) / np.sum(pred_near_zeros) if np.sum(pred_near_zeros) > 0 else 0
            recall = np.sum(true_zeros & pred_near_zeros) / np.sum(true_zeros)
            print(f"Sparsity Precision: {precision:.1%}")
            print(f"Sparsity Recall: {recall:.1%}")
        
        return {
            'predictions': predictions,
            'targets': targets,
            'inputs': inputs,
            'correlation': corr,
            'level_performance': level_performance
        }
    
    def visualize_attention_patterns(self, num_samples: int = 5, save_path: str = None):
        """Visualize attention patterns to understand what the model learned."""
        
        # Generate test samples
        seq_len = self.wavelet_prior.total_coeffs
        test_dataset = WaveletDataset(self.wavelet_prior, num_samples, seq_len)
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(15, 3*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample in enumerate(test_dataset):
            x = sample['x'].unsqueeze(0).to(self.device)  # Add batch dimension
            target = sample['log_prob'].numpy()
            
            # Get prediction
            with torch.no_grad():
                prediction = self.model(x).cpu().numpy().flatten()
            
            # Input vs Target vs Prediction
            axes[i, 0].plot(target, label='True Coefficients', linewidth=2)
            axes[i, 0].plot(x.cpu().numpy().flatten(), label='Noisy Input', alpha=0.7, linewidth=1)
            axes[i, 0].plot(prediction, label='Predicted', linewidth=2, linestyle='--')
            axes[i, 0].set_title(f'Sample {i+1}: Coefficient Recovery')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Residual analysis
            residual = prediction - target
            axes[i, 1].plot(residual, label='Prediction Error', linewidth=2)
            axes[i, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
            axes[i, 1].set_title(f'Sample {i+1}: Prediction Errors')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_configurations(self, other_analyzers: List['WaveletAnalyzer'], 
                             labels: List[str], save_path: str = None):
        """Compare performance across different model configurations."""
        
        all_analyzers = [self] + other_analyzers
        all_labels = ['Current'] + labels
        
        # Collect performance metrics
        results = []
        for analyzer, label in zip(all_analyzers, all_labels):
            if analyzer.training_history:
                final_train = analyzer.training_history['train_losses'][-1]
                final_val = analyzer.training_history['val_losses'][-1]
                
                results.append({
                    'label': label,
                    'train_loss': final_train['loss'],
                    'val_loss': final_val['loss'],
                    'correlation': final_val.get('correlation', 0),
                    'r_squared': final_val.get('r_squared', 0)
                })
        
        if not results:
            print("No training history available for comparison")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['train_loss', 'val_loss', 'correlation', 'r_squared']
        titles = ['Training Loss', 'Validation Loss', 'Correlation', 'R-squared']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx//2, idx%2]
            
            labels = [r['label'] for r in results]
            values = [r[metric] for r in results]
            
            bars = ax.bar(labels, values, alpha=0.7)
            ax.set_title(title)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, output_dir: str = "analysis_results"):
        """Generate a comprehensive analysis report."""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating comprehensive analysis report...")
        
        # 1. Training history
        self.plot_training_history(save_path=f"{output_dir}/training_history.png")
        
        # 2. Prediction analysis
        pred_results = self.analyze_predictions(save_path=f"{output_dir}/prediction_analysis.png")
        
        # 3. Attention patterns
        self.visualize_attention_patterns(save_path=f"{output_dir}/attention_patterns.png")
        
        # 4. Save detailed results
        detailed_results = {
            'model_config': self.config,
            'wavelet_config': self.wavelet_prior.get_info(),
            'training_history': self.training_history,
            'prediction_results': pred_results
        }
        
        with open(f"{output_dir}/detailed_results.pkl", 'wb') as f:
            pickle.dump(detailed_results, f)
        
        # 5. Generate summary report
        with open(f"{output_dir}/summary_report.txt", 'w') as f:
            f.write("WAVELET TRANSFORMER ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write("MODEL CONFIGURATION:\n")
            f.write("-"*20 + "\n")
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nWAVELET CONFIGURATION:\n")
            f.write("-"*20 + "\n")
            wavelet_info = self.wavelet_prior.get_info()
            for key, value in wavelet_info.items():
                if key != 'sparsity_weights':
                    f.write(f"{key}: {value}\n")
            
            f.write("\nSPARSITY STRUCTURE:\n")
            f.write("-"*20 + "\n")
            for level, weight in wavelet_info['sparsity_weights'].items():
                f.write(f"Level {level}: {weight:.1%} non-zero probability\n")
            
            if self.training_history:
                final_val = self.training_history['val_losses'][-1]
                f.write("\nFINAL PERFORMANCE:\n")
                f.write("-"*20 + "\n")
                f.write(f"Validation Loss: {final_val['loss']:.4f}\n")
                f.write(f"R-squared: {final_val.get('r_squared', 0):.3f}\n")
                f.write(f"Correlation: {final_val.get('correlation', 0):.3f}\n")
        
        print(f"Analysis report generated in {output_dir}/")
        print(f"Files created:")
        print(f"  - training_history.png")
        print(f"  - prediction_analysis.png") 
        print(f"  - attention_patterns.png")
        print(f"  - detailed_results.pkl")
        print(f"  - summary_report.txt")


def load_and_analyze(model_path: str, output_dir: str = "analysis_results"):
    """Convenience function to load model and run full analysis."""
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and analyze
    analyzer = WaveletAnalyzer(model_path, device=device)
    analyzer.generate_report(output_dir)
    
    return analyzer


if __name__ == "__main__":
    # Example usage
    model_path = "checkpoints/naive_wavelet_transformer.pkl"
    
    if os.path.exists(model_path):
        print("Loading and analyzing saved model...")
        analyzer = load_and_analyze(model_path)
    else:
        print(f"Model file not found: {model_path}")
        print("Please train a model first using train/train_wavelets.py") 