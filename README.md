# Bayesian Transformer Framework (BTF)

A simple PyTorch framework for training transformers on synthetic data generated from user-defined priors to solve nonparametric Bayesian problems empirically.

## Overview

This framework provides a naive but functional implementation of a Bayesian transformer that:
- Generates synthetic data using customizable priors
- Trains a standard transformer architecture with single attention head
- Uses standard batching procedures with Adam optimizer
- Evaluates performance on nonparametric Bayesian inference tasks

## Repository Structure

```
BTF/
├── data/
│   └── data_generator.py    # Synthetic data generation with priors
├── models/
│   └── transformer.py      # Transformer architecture implementation
├── train/
│   └── training.py         # Training loop and optimization
├── test/
│   └── test.py            # Unit tests and performance evaluation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BTF
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Synthetic Data

```python
from data.data_generator import create_data_generator

# Create a custom prior class
class MyPrior:
    def __init__(self):
        self.dim = 1
        
    def sample(self, batch_size, seq_len):
        # Your custom sampling logic here
        return torch.randn(batch_size, seq_len, self.dim)
        
    def log_prob(self, x):
        # Your custom log probability computation here
        return -0.5 * (x ** 2 + np.log(2 * np.pi))

# Create data generator with your prior
prior = MyPrior()
generator = create_data_generator(prior)

# Generate a batch of data
batch = generator.generate_batch(batch_size=32, seq_len=50)
print(f"Data shape: {batch['x'].shape}")
print(f"Log probabilities shape: {batch['log_prob'].shape}")
```

### 2. Create and Train a Model

```python
from models.transformer import create_model
from train.training import Trainer

# Create transformer model
model = create_model(
    input_dim=1,
    d_model=128,
    n_heads=1,
    n_layers=4,
    output_dim=1
)

# Create trainer
trainer = Trainer(model, generator, device='cpu')

# Train the model
history = trainer.train(
    num_epochs=50,
    train_samples=8000,
    val_samples=2000,
    batch_size=32,
    seq_len=50
)
```

### 3. Run the Complete Training Pipeline

```python
# Run the main training script
python train/training.py
```

This will:
- Create a data generator with a simple test prior
- Initialize a transformer model
- Train for 50 epochs
- Save the best model as `best_model.pt`
- Generate training curves plot

## Creating Custom Priors

You need to create your own prior classes that inherit from the `Prior` abstract base class:

```python
from data.data_generator import Prior

class MyCustomPrior(Prior):
    def __init__(self, **kwargs):
        # Initialize your prior parameters
        self.dim = kwargs.get('dim', 1)
        
    def sample(self, batch_size: int, seq_len: int) -> torch.Tensor:
        # Implement your sampling logic
        return torch.randn(batch_size, seq_len, self.dim)
        
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        # Implement your log probability computation
        return -0.5 * (x ** 2 + np.log(2 * np.pi))
```

## Model Configuration

The transformer model supports the following parameters:

- `input_dim`: Dimension of input features
- `d_model`: Model dimension (default: 256)
- `n_heads`: Number of attention heads (default: 1)
- `n_layers`: Number of transformer layers (default: 6)
- `d_ff`: Feed-forward dimension (default: 1024)
- `max_seq_len`: Maximum sequence length (default: 1000)
- `dropout`: Dropout probability (default: 0.1)
- `output_dim`: Output dimension (default: 1)

## Testing

Run the comprehensive test suite:

```bash
python test/test.py
```

This will:
- Run unit tests for all components
- Perform integration tests
- Run performance comparisons between different priors
- Generate visualization plots

## Training Configuration

Default training parameters in `train/training.py`:

```python
config = {
    'num_epochs': 50,
    'train_samples': 8000,
    'val_samples': 2000,
    'batch_size': 32,
    'seq_len': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

## Extending the Framework

### Customizing the Model

The transformer architecture can be modified by editing `models/transformer.py`. Key components:

- `BayesianTransformer`: Main model class
- `TransformerBlock`: Individual transformer layers
- `MultiHeadAttention`: Attention mechanism
- `FeedForward`: Feed-forward networks

## Output Files

The framework generates several output files:

- `best_model.pt`: Best trained model checkpoint
- `training_curves.png`: Training and validation loss curves
- `performance_test_results.png`: Performance comparison plots

## Requirements

- Python 3.7+
- PyTorch 2.0+
- NumPy
- Matplotlib
- SciPy
- tqdm

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## Future Enhancements

- Example implementations of common priors (GP, Dirichlet Process, etc.)
- Multi-head attention variants
- Different loss functions for Bayesian inference
- Uncertainty quantification metrics
- Real-world dataset integration
- Distributed training support
