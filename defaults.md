# Default Parameters

## Transformer Model

| Parameter | Value | Description |
|-----------|-------|-------------|
| `input_dim` | 1 | Input feature dimension |
| `d_model` | 256 | Model embedding dimension |
| `n_heads` | 1 | Number of attention heads |
| `n_layers` | 3 | Number of transformer layers |
| `d_ff` | 1024 | Feed-forward network dimension |
| `max_seq_len` | 1000 | Maximum sequence length |
| `dropout` | 0.1 | Dropout probability |
| `output_dim` | 1 | Output dimension |

## Training Configuration (Actual Usage)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 128 | Reduced model dimension for faster training |
| `n_heads` | 1 | Single attention head |
| `n_layers` | 6 | Standard transformer depth |
| `d_ff` | 512 | Reduced feed-forward dimension |
| `dropout` | 0.1 | Standard dropout rate |

## Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_epochs` | 50 | Training epochs |
| `train_samples` | 3000 | Training dataset size |
| `val_samples` | 500 | Validation dataset size |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 1e-4 | Adam optimizer learning rate |
| `weight_decay` | 1e-5 | L2 regularization weight |

## Data Generation

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n` | 128 | Sample size for noise level |
| `max_resolution` | 4 | Maximum wavelet resolution level |
| `configuration` | 'standard' | Wavelet prior configuration type |

## Component Defaults

### TransformerBlock
| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_ff` | 2048 | Default feed-forward dimension (overridden) |
| `dropout` | 0.1 | Layer dropout rate |

### MultiHeadAttention
| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_heads` | 1 | Default attention heads |
| `dropout` | 0.1 | Attention dropout rate |

### PositionalEncoding
| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_len` | 5000 | Maximum positional encoding length |

### WaveletTrainer
| Parameter | Value | Description |
|-----------|-------|-------------|
| `device` | 'cpu' | Default device (auto-detected) |
| `learning_rate` | 1e-4 | Default learning rate |
| `weight_decay` | 1e-5 | Default weight decay | 