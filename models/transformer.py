import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer inputs."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # self.pe shape: (max_len, 1, d_model)
        # Extract (seq_len, d_model) and add batch dimension
        return x + self.pe[:seq_len, 0, :].unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for multi-head attention."""
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_output, self.attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attention_output)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                   V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention."""
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V), attention_weights


class FeedForward(nn.Module):
    """Feed-forward network used in transformer blocks."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feed-forward network."""
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward."""
    
    def __init__(self, d_model: int, n_heads: int = 1, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer block."""
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class BayesianTransformer(nn.Module):
    """Simple Bayesian Transformer for learning priors."""
    
    def __init__(self, 
                 input_dim: int = 1,
                 d_model: int = 256,
                 n_heads: int = 1,
                 n_layers: int = 6,
                 d_ff: int = 1024,
                 max_seq_len: int = 1000,
                 dropout: float = 0.1,
                 output_dim: int = 1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection to log probability
        self.output_projection = nn.Linear(d_model, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Log probability predictions of shape (batch_size, seq_len, output_dim)
        """
        # Input projection and positional encoding
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Output projection
        log_prob = self.output_projection(x)  # (batch_size, seq_len, output_dim)
        
        return log_prob
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss between predictions and targets.
        
        Args:
            predictions: Predicted log probabilities
            targets: Target log probabilities
            
        Returns:
            Mean squared error loss
        """
        return F.mse_loss(predictions, targets)
    
    def get_attention_weights(self) -> list:
        """Get attention weights from all transformer blocks."""
        attention_weights = []
        for block in self.transformer_blocks:
            if hasattr(block.attention, 'attention_weights') and block.attention.attention_weights is not None:
                attention_weights.append(block.attention.attention_weights)
        return attention_weights


def create_model(input_dim: int = 1, 
                d_model: int = 256, 
                n_heads: int = 1, 
                n_layers: int = 6, 
                d_ff: int = 1024, 
                max_seq_len: int = 1000,
                dropout: float = 0.1,
                output_dim: int = 1) -> BayesianTransformer:
    """Factory function to create a Bayesian Transformer model.
    
    Args:
        input_dim: Dimension of input features
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        output_dim: Output dimension (usually 1 for log probability)
        
    Returns:
        BayesianTransformer model
    """
    return BayesianTransformer(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        output_dim=output_dim
    )


# Example usage
if __name__ == "__main__":
    # Test the model
    print("Testing Bayesian Transformer...")
    
    # Create model
    model = create_model(input_dim=2, d_model=128, n_heads=1, n_layers=4)
    
    # Test forward pass
    batch_size, seq_len, input_dim = 4, 10, 2
    x = torch.randn(batch_size, seq_len, input_dim)
    
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
        # Test loss computation
        targets = torch.randn_like(output)
        loss = model.compute_loss(output, targets)
        print(f"Loss: {loss.item():.4f}")
        
        # Check attention weights
        attention_weights = model.get_attention_weights()
        print(f"Number of attention weight matrices: {len(attention_weights)}")
        if attention_weights:
            print(f"First attention weight shape: {attention_weights[0].shape}")
    
    print("Model test completed successfully!") 