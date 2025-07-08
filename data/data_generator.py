import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod


class Prior(ABC):
    """Abstract base class for defining priors."""
    
    @abstractmethod
    def sample(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Generate samples from the prior."""
        pass
    
    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of samples under the prior."""
        pass



class SyntheticDataGenerator:
    """Generator for synthetic data using different priors."""
    
    def __init__(self, prior: Prior, device: str = 'cpu'):
        self.prior = prior
        self.device = device
    
    def generate_batch(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """Generate a batch of synthetic data.
        
        Args:
            batch_size: Number of sequences in batch
            seq_len: Length of each sequence
            
        Returns:
            Dictionary containing 'x' (input data) and 'log_prob' (target log probabilities)
        """
        # Generate samples from prior
        x = self.prior.sample(batch_size, seq_len).to(self.device)
        
        # Compute log probabilities (these serve as targets)
        log_prob = self.prior.log_prob(x).to(self.device)
        
        return {
            'x': x,
            'log_prob': log_prob
        }
    
    def generate_dataset(self, num_batches: int, batch_size: int, seq_len: int) -> list:
        """Generate multiple batches of synthetic data.
        
        Args:
            num_batches: Number of batches to generate
            batch_size: Size of each batch
            seq_len: Length of each sequence
            
        Returns:
            List of dictionaries, each containing a batch of data
        """
        dataset = []
        for _ in range(num_batches):
            batch = self.generate_batch(batch_size, seq_len)
            dataset.append(batch)
        
        return dataset


def create_data_generator(prior: Prior) -> SyntheticDataGenerator:
    """Factory function to create data generators with a prior.
    
    Args:
        prior: A Prior instance
        
    Returns:
        SyntheticDataGenerator instance
    """
    return SyntheticDataGenerator(prior)


# Example usage
if __name__ == "__main__":
    # Users should define their own prior classes that inherit from Prior
    # and pass them to create_data_generator
    print("To use this module, create a class that inherits from Prior and pass it to create_data_generator()") 