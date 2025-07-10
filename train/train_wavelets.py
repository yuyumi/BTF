import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import json
from typing import Dict, List, Tuple

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load config
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

from data.wavelet_data_generator import create_wavelet_prior, WaveletSpikeSlabPrior
from models.transformer import create_model, BayesianTransformer


class WaveletDataset(Dataset):
    """PyTorch Dataset for wavelet spike-slab data."""
    
    def __init__(self, wavelet_prior: WaveletSpikeSlabPrior, num_samples: int, seq_len: int):
        self.wavelet_prior = wavelet_prior
        self.num_samples = num_samples
        self.seq_len = seq_len
        
        # Pre-generate data for reproducibility
        print(f"Generating {num_samples} wavelet samples...")
        self.data = []
        
        for _ in tqdm(range(num_samples), desc="Generating data"):
            # Generate training batch (batch_size=1 for individual samples)
            batch = wavelet_prior.generate_training_batch(1, seq_len)
            self.data.append({
                'x': batch['x'][0],           # Noisy observations Y_{j,k}
                'log_prob': batch['log_prob'][0]  # True coefficients θ_{j,k}
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class WaveletTrainer:
    """Standard trainer for wavelet data - no domain-specific modifications."""
    
    def __init__(self, 
                 model: BayesianTransformer,
                 wavelet_prior: WaveletSpikeSlabPrior,
                 device: str = 'cpu',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.wavelet_prior = wavelet_prior
        self.device = device
        
        # Standard optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Standard MSE loss - let transformer learn Bayesian structure naturally
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute basic metrics - no sparsity-specific measures."""
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        # Standard regression metrics
        mse = np.mean((predictions_np - targets_np) ** 2)
        mae = np.mean(np.abs(predictions_np - targets_np))
        
        # Correlation between predictions and targets
        if predictions_np.size > 1:
            correlation = np.corrcoef(predictions_np.flatten(), targets_np.flatten())[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # R-squared
        if np.var(targets_np) > 0:
            r_squared = 1 - (np.var(predictions_np - targets_np) / np.var(targets_np))
        else:
            r_squared = 0.0
        
        # Simple sparsity observations (for analysis, not optimization)
        true_sparsity = np.mean(np.abs(targets_np) < 1e-6)
        pred_sparsity = np.mean(np.abs(predictions_np) < 1e-6)
        
        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'r_squared': r_squared,
            'true_sparsity': true_sparsity,
            'pred_sparsity': pred_sparsity
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch - standard training loop."""
        self.model.train()
        total_loss = 0.0
        total_metrics = {}
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            # Move data to device
            x = batch['x'].to(self.device)
            targets = batch['log_prob'].to(self.device)
            
            # Standard forward pass
            self.optimizer.zero_grad()
            predictions = self.model(x)
            
            # Standard MSE loss
            loss = self.criterion(predictions, targets)
            
            # Standard backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Compute metrics for monitoring
            metrics = self.compute_metrics(predictions, targets)
            for key, value in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value
            
            num_batches += 1
            
            # Update progress bar with basic info
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'corr': f'{metrics["correlation"]:.3f}'
            })
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch - standard validation loop."""
        self.model.eval()
        total_loss = 0.0
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                x = batch['x'].to(self.device)
                targets = batch['log_prob'].to(self.device)
                
                # Forward pass
                predictions = self.model(x)
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                
                # Compute metrics
                metrics = self.compute_metrics(predictions, targets)
                for key, value in metrics.items():
                    total_metrics[key] = total_metrics.get(key, 0.0) + value
                
                num_batches += 1
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def train(self, 
              num_epochs: int = 100,
              train_samples: int = 5000,
              val_samples: int = 1000,
              batch_size: int = 32,
              save_path: str = None) -> Dict:
        """Train the model on wavelet data - standard training procedure."""
        
        seq_len = self.wavelet_prior.total_coeffs
        
        print(f"Training Standard Transformer on Wavelet Spike-Slab Data")
        print(f"Goal: Test if naive transformer can learn nonparametric Bayesian structure")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Sequence length: {seq_len}")
        print(f"Loss: Standard MSE (no domain-specific modifications)")
        
        # Create datasets
        train_dataset = WaveletDataset(self.wavelet_prior, train_samples, seq_len)
        val_dataset = WaveletDataset(self.wavelet_prior, val_samples, seq_len)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_results = self.train_epoch(train_loader)
            
            # Validate
            val_results = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_results['loss'])
            
            # Save results
            self.train_losses.append(train_results)
            self.val_losses.append(val_results)
            
            # Print epoch results - focus on standard metrics
            print(f"Train - Loss: {train_results['loss']:.4f}, "
                  f"R²: {train_results['r_squared']:.3f}, "
                  f"Corr: {train_results['correlation']:.3f}")
            print(f"Val   - Loss: {val_results['loss']:.4f}, "
                  f"R²: {val_results['r_squared']:.3f}, "
                  f"Sparsity: T={val_results['true_sparsity']:.2%}/P={val_results['pred_sparsity']:.2%}")
            
            # Save best model
            if val_results['loss'] < self.best_val_loss:
                self.best_val_loss = val_results['loss']
                if save_path:
                    self.save_model(save_path)
                    # Also save as pickle for easy analysis
                    pickle_path = save_path.replace('.pth', '.pkl')
                    self.save_model_pickle(pickle_path)
                    print(f"New best model saved to {save_path}")
                    print(f"Pickle file saved to {pickle_path}")
        
        # Training completed
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'wavelet_config': self.wavelet_prior.get_info(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def save_model_pickle(self, path: str):
        """Save complete model as pickle file for easy analysis."""
        import pickle
        
        # Save everything needed for analysis
        save_data = {
            'model': self.model.cpu(),  # Move to CPU for portability
            'wavelet_prior': self.wavelet_prior,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss
            },
            'config': {
                'model_params': sum(p.numel() for p in self.model.parameters()),
                'device': self.device,
                'seq_len': self.wavelet_prior.total_coeffs
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Move model back to original device
        self.model.to(self.device)
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        return checkpoint.get('wavelet_config', {})
    
    def evaluate_sparsity_learning(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate how well the transformer learned the sparsity structure."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['x'].to(self.device)
                targets = batch['log_prob'].to(self.device)
                predictions = self.model(x)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Analyze sparsity learning
        true_zeros = np.abs(targets) < 1e-6
        pred_near_zeros = np.abs(predictions) < 0.1  # Threshold for "learned sparsity"
        
        if np.sum(true_zeros) > 0:
            sparsity_precision = np.sum(true_zeros & pred_near_zeros) / np.sum(pred_near_zeros) if np.sum(pred_near_zeros) > 0 else 0
            sparsity_recall = np.sum(true_zeros & pred_near_zeros) / np.sum(true_zeros)
        else:
            sparsity_precision = sparsity_recall = 0
        
        # Coefficient magnitude analysis
        nonzero_mask = ~true_zeros
        if np.sum(nonzero_mask) > 0:
            nonzero_correlation = np.corrcoef(
                predictions[nonzero_mask].flatten(), 
                targets[nonzero_mask].flatten()
            )[0, 1] if np.sum(nonzero_mask) > 1 else 0
        else:
            nonzero_correlation = 0
        
        return {
            'sparsity_precision': sparsity_precision,
            'sparsity_recall': sparsity_recall,
            'nonzero_correlation': nonzero_correlation,
            'learned_sparsity': np.mean(pred_near_zeros),
            'true_sparsity': np.mean(true_zeros)
        }


def main():
    """Main training script - test if naive transformer can learn Bayesian structure."""
    
    # Load configuration
    config = load_config()
    data_config = config.get('data_generation', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    # Set device with detailed feedback
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
        print("CUDA not available - using CPU")
        print("For faster training, install CUDA-enabled PyTorch")
    
    # Create wavelet prior
    print("Creating wavelet spike-slab prior...")
    wavelet_prior = create_wavelet_prior(
        data_config.get('configuration', 'standard'),
        n=data_config.get('n', 128),
        max_resolution=data_config.get('max_resolution', 4)
    )
    
    # Print configuration
    info = wavelet_prior.get_info()
    print(f"\nWavelet configuration:")
    for key, value in info.items():
        if key != 'sparsity_weights':
            print(f"  {key}: {value}")
    
    # Show expected sparsity
    print(f"\nExpected sparsity structure:")
    for j, weight in info['sparsity_weights'].items():
        print(f"  Level {j}: {weight:.1%} non-zero probability")
    
    # Create standard transformer model
    seq_len = wavelet_prior.total_coeffs
    model = create_model(
        input_dim=model_config.get('input_dim', 1),
        d_model=model_config.get('d_model', 128),
        n_heads=model_config.get('n_heads', 1),
        n_layers=model_config.get('n_layers', 6),
        d_ff=model_config.get('d_ff', 512),
        max_seq_len=seq_len,
        dropout=model_config.get('dropout', 0.1),
        output_dim=model_config.get('output_dim', 1)
    )
    
    print(f"\nStandard transformer created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Architecture: {model.n_layers} layers, {model.d_model} dim, {model.transformer_blocks[0].attention.n_heads} head(s)")
    
    # Create trainer - no domain-specific modifications
    trainer = WaveletTrainer(
        model=model,
        wavelet_prior=wavelet_prior,
        device=device,
        learning_rate=training_config.get('learning_rate', 1e-4),
        weight_decay=training_config.get('weight_decay', 1e-5)
    )
    
    # Train the model
    print(f"\nStarting training...")
    results = trainer.train(
        num_epochs=training_config.get('num_epochs', 50),
        train_samples=training_config.get('train_samples', 3000),
        val_samples=training_config.get('val_samples', 500),
        batch_size=training_config.get('batch_size', 32),
        save_path='checkpoints/naive_wavelet_transformer.pth'
    )
    
    # Evaluate sparsity learning
    print(f"\nEvaluating learned Bayesian structure...")
    val_dataset = WaveletDataset(wavelet_prior, 500, seq_len)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    sparsity_results = trainer.evaluate_sparsity_learning(val_loader)
    print(f"Sparsity Learning Results:")
    print(f"  True sparsity: {sparsity_results['true_sparsity']:.1%}")
    print(f"  Learned sparsity: {sparsity_results['learned_sparsity']:.1%}")
    print(f"  Sparsity precision: {sparsity_results['sparsity_precision']:.1%}")
    print(f"  Sparsity recall: {sparsity_results['sparsity_recall']:.1%}")
    print(f"  Non-zero correlation: {sparsity_results['nonzero_correlation']:.3f}")
    
    print("\n" + "="*60)
    print("EXPERIMENT CONCLUSION:")
    print("Can a naive transformer learn nonparametric Bayesian structure?")
    print(f"Answer: {'YES' if sparsity_results['sparsity_precision'] > 0.7 else 'PARTIAL' if sparsity_results['sparsity_precision'] > 0.3 else 'NO'}")
    print("="*60)
    
    return trainer, results


if __name__ == "__main__":
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Run training experiment
    trainer, results = main() 