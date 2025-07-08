import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import unittest
from typing import Dict, Any

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_generator import create_data_generator
from models.transformer import create_model, BayesianTransformer
from train.training import Trainer, SyntheticDataset


class TestDataGenerator(unittest.TestCase):
    """Test the data generator components."""
    
    def test_data_generator(self):
        """Test the data generator factory."""
        print("Testing Data Generator Factory...")
        
        # Create a simple test prior
        class TestPrior:
            def __init__(self):
                self.dim = 1
                
            def sample(self, batch_size, seq_len):
                return torch.randn(batch_size, seq_len, self.dim)
                
            def log_prob(self, x):
                return -0.5 * (x ** 2 + np.log(2 * np.pi))
        
        prior = TestPrior()
        generator = create_data_generator(prior)
        batch = generator.generate_batch(batch_size=5, seq_len=10)
        
        self.assertIn('x', batch)
        self.assertIn('log_prob', batch)
        self.assertEqual(batch['x'].shape, (5, 10, 1))
        self.assertEqual(batch['log_prob'].shape, (5, 10, 1))
        
        print("  Data generator factory tests passed")


class TestTransformer(unittest.TestCase):
    """Test the transformer model components."""
    
    def test_model_creation(self):
        """Test model creation and basic functionality."""
        print("Testing Transformer Model Creation...")
        
        model = create_model(
            input_dim=2,
            d_model=128,
            n_heads=1,
            n_layers=3,
            output_dim=1
        )
        
        # Check model structure
        self.assertIsInstance(model, BayesianTransformer)
        self.assertEqual(model.input_dim, 2)
        self.assertEqual(model.d_model, 128)
        self.assertEqual(model.n_layers, 3)
        
        print("  Model creation tests passed")
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        print("Testing Forward Pass...")
        
        model = create_model(input_dim=1, d_model=64, n_layers=2)
        
        # Test forward pass
        batch_size, seq_len, input_dim = 4, 10, 1
        x = torch.randn(batch_size, seq_len, input_dim)
        
        with torch.no_grad():
            output = model(x)
            
        self.assertEqual(output.shape, (batch_size, seq_len, 1))
        self.assertTrue(torch.all(torch.isfinite(output)))
        
        print("  Forward pass tests passed")
    
    def test_loss_computation(self):
        """Test loss computation."""
        print("Testing Loss Computation...")
        
        model = create_model(input_dim=1, d_model=64, n_layers=2)
        
        # Create dummy data
        batch_size, seq_len = 4, 10
        predictions = torch.randn(batch_size, seq_len, 1)
        targets = torch.randn(batch_size, seq_len, 1)
        
        loss = model.compute_loss(predictions, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss))
        
        print("  Loss computation tests passed")


class TestTraining(unittest.TestCase):
    """Test the training components."""
    
    def test_dataset_creation(self):
        """Test synthetic dataset creation."""
        print("Testing Dataset Creation...")
        
        # Create a simple test prior
        class TestPrior:
            def __init__(self):
                self.dim = 1
                
            def sample(self, batch_size, seq_len):
                return torch.randn(batch_size, seq_len, self.dim)
                
            def log_prob(self, x):
                return -0.5 * (x ** 2 + np.log(2 * np.pi))
        
        prior = TestPrior()
        generator = create_data_generator(prior)
        
        # Create dataset
        dataset = SyntheticDataset(generator, num_samples=100, batch_size=10, seq_len=20)
        
        self.assertEqual(len(dataset), 100)
        
        # Test data loading
        sample = dataset[0]
        self.assertIn('x', sample)
        self.assertIn('log_prob', sample)
        self.assertEqual(sample['x'].shape, (20, 1))
        self.assertEqual(sample['log_prob'].shape, (20, 1))
        
        print("  Dataset creation tests passed")
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        print("Testing Trainer Creation...")
        
        # Create a simple test prior
        class TestPrior:
            def __init__(self):
                self.dim = 1
                
            def sample(self, batch_size, seq_len):
                return torch.randn(batch_size, seq_len, self.dim)
                
            def log_prob(self, x):
                return -0.5 * (x ** 2 + np.log(2 * np.pi))
        
        prior = TestPrior()
        generator = create_data_generator(prior)
        model = create_model(input_dim=1, d_model=64, n_layers=2)
        
        # Create trainer
        trainer = Trainer(model, generator, device='cpu')
        
        self.assertIsInstance(trainer.model, BayesianTransformer)
        self.assertEqual(trainer.device, 'cpu')
        self.assertIsInstance(trainer.optimizer, torch.optim.Adam)
        
        print("  Trainer creation tests passed")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete framework."""
    
    def test_end_to_end_training(self):
        """Test a complete end-to-end training run."""
        print("Testing End-to-End Training...")
        
        # Create a simple test prior
        class TestPrior:
            def __init__(self):
                self.dim = 1
                
            def sample(self, batch_size, seq_len):
                return torch.randn(batch_size, seq_len, self.dim)
                
            def log_prob(self, x):
                return -0.5 * (x ** 2 + np.log(2 * np.pi))
        
        prior = TestPrior()
        generator = create_data_generator(prior)
        model = create_model(input_dim=1, d_model=64, n_layers=2)
        trainer = Trainer(model, generator, device='cpu')
        
        # Run short training
        history = trainer.train(
            num_epochs=2,
            train_samples=100,
            val_samples=50,
            batch_size=10,
            seq_len=20
        )
        
        # Check training results
        self.assertIn('train_losses', history)
        self.assertIn('val_losses', history)
        self.assertEqual(len(history['train_losses']), 2)
        self.assertEqual(len(history['val_losses']), 2)
        
        print("  End-to-end training tests passed")


def run_performance_test():
    """Run performance tests and visualizations."""
    print("\n" + "="*50)
    print("PERFORMANCE TESTS")
    print("="*50)
    
    # Test a simple prior
    class TestPrior:
        def __init__(self):
            self.dim = 1
            
        def sample(self, batch_size, seq_len):
            return torch.randn(batch_size, seq_len, self.dim)
            
        def log_prob(self, x):
            return -0.5 * (x ** 2 + np.log(2 * np.pi))
    
    prior = TestPrior()
    generator = create_data_generator(prior)
    model = create_model(input_dim=1, d_model=64, n_layers=3)
    trainer = Trainer(model, generator, device='cpu', learning_rate=1e-3)
    
    # Train for a few epochs
    history = trainer.train(
        num_epochs=10,
        train_samples=500,
        val_samples=100,
        batch_size=16,
        seq_len=30
    )
    
    # Test model predictions
    test_batch = generator.generate_batch(batch_size=5, seq_len=30)
    with torch.no_grad():
        predictions = model(test_batch['x'])
        test_loss = model.compute_loss(predictions, test_batch['log_prob'])
    
    print(f"  Final validation loss: {history['val_losses'][-1]:.4f}")
    print(f"  Test loss: {test_loss.item():.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='train', alpha=0.7)
    plt.plot(history['val_losses'], label='val', alpha=0.7, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Generate and plot sample data
    batch = generator.generate_batch(batch_size=1, seq_len=200)
    data = batch['x'][0, :, 0].numpy()
    plt.plot(data, label='samples', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Sample Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nPerformance test completed!")
    print("Results saved to 'performance_test_results.png'")


def run_all_tests():
    """Run all unit tests."""
    print("="*50)
    print("RUNNING UNIT TESTS")
    print("="*50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataGenerator))
    test_suite.addTest(unittest.makeSuite(TestTransformer))
    test_suite.addTest(unittest.makeSuite(TestTraining))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    if result.wasSuccessful():
        print("\nAll tests passed!")
        return True
    else:
        print(f"\n{len(result.failures)} test(s) failed")
        return False


def main():
    """Main test function."""
    print("Bayesian Transformer Framework - Test Suite")
    print("="*50)
    
    # Run unit tests
    tests_passed = run_all_tests()
    
    if tests_passed:
        # Run performance tests if unit tests pass
        run_performance_test()
    else:
        print("Skipping performance tests due to failed unit tests.")


if __name__ == "__main__":
    main() 