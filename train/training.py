import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_generator import create_data_generator, SyntheticDataGenerator
from models.transformer import create_model, BayesianTransformer


class SyntheticDataset(Dataset):
    """PyTorch Dataset wrapper for synthetic data."""
    
    def __init__(self, data_generator: SyntheticDataGenerator, num_samples: int, 
                 batch_size: int, seq_len: int):
        self.data_generator = data_generator
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        # Pre-generate data
        self.data = []
        num_batches = num_samples // batch_size
        for _ in range(num_batches):
            batch = data_generator.generate_batch(batch_size, seq_len)
            # Split batch into individual samples
            for i in range(batch_size):
                self.data.append({
                    'x': batch['x'][i],
                    'log_prob': batch['log_prob'][i]
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class Trainer:
    """Trainer class for the Bayesian Transformer."""
    
    def __init__(self, 
                 model: BayesianTransformer,
                 data_generator: SyntheticDataGenerator,
                 device: str = 'cpu',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.data_generator = data_generator
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            # Move data to device
            x = batch['x'].to(self.device)
            targets = batch['log_prob'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(x)
            loss = self.model.compute_loss(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate_epoch(self, dataloader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                x = batch['x'].to(self.device)
                targets = batch['log_prob'].to(self.device)
                
                # Forward pass
                predictions = self.model(x)
                loss = self.model.compute_loss(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, 
              num_epochs: int = 100,
              train_samples: int = 10000,
              val_samples: int = 2000,
              batch_size: int = 32,
              seq_len: int = 50,
              save_path: str = None) -> dict:
        """Train the model."""
        
        print(f"Training Bayesian Transformer for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create datasets
        train_dataset = SyntheticDataset(self.data_generator, train_samples, batch_size, seq_len)
        val_dataset = SyntheticDataset(self.data_generator, val_samples, batch_size, seq_len)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if save_path:
                    self.save_model(save_path)
                    print(f"New best model saved to {save_path}")
        
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
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training and validation curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()


def main():
    """Main training function."""
    
    # Configuration
    config = {        
        # Model parameters
        'input_dim': 1,
        'd_model': 128,
        'n_heads': 1,
        'n_layers': 4,
        'd_ff': 512,
        'max_seq_len': 200,
        'dropout': 0.1,
        'output_dim': 1,
        
        # Training parameters
        'num_epochs': 50,
        'train_samples': 8000,
        'val_samples': 2000,
        'batch_size': 32,
        'seq_len': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create a simple test prior
    class TestPrior:
        def __init__(self):
            self.dim = 1
            
        def sample(self, batch_size, seq_len):
            return torch.randn(batch_size, seq_len, self.dim)
            
        def log_prob(self, x):
            return -0.5 * (x ** 2 + np.log(2 * np.pi))
    
    prior = TestPrior()
    data_generator = create_data_generator(prior)
    
    # Create model
    model = create_model(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        output_dim=config['output_dim']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        data_generator=data_generator,
        device=config['device'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Train model
    history = trainer.train(
        num_epochs=config['num_epochs'],
        train_samples=config['train_samples'],
        val_samples=config['val_samples'],
        batch_size=config['batch_size'],
        seq_len=config['seq_len'],
        save_path='best_model.pt'
    )
    
    # Plot training curves
    trainer.plot_training_curves('training_curves.png')
    
    return trainer, history


if __name__ == "__main__":
    trainer, history = main() 