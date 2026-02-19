"""
Transformer-based Sentiment Classification Model

This module contains the implementation of a custom Transformer architecture
for sentiment classification on IMDB movie reviews.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module to inject position information into embeddings.
    Uses sinusoidal positional encodings as described in "Attention is All You Need".
    """
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    Allows the model to jointly attend to information from different representation subspaces.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape to (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Applies two linear transformations with a GELU activation in between.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)  # Using GELU activation (common in modern transformers)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block.
    Consists of Multi-Head Attention followed by Feed-Forward Network,
    with residual connections and layer normalization.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection and layer norm
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerClassifier(nn.Module):
    """
    Transformer-based Text Classification Model.
    
    Architecture:
    1. Embedding Layer: Converts token indices to dense vectors
    2. Positional Encoding: Adds position information
    3. Transformer Encoder Blocks: Stack of encoder layers for context understanding
    4. Global Average Pooling: Aggregates sequence information
    5. Classification Head: MLP for binary classification
    
    Why this architecture:
    - Self-attention allows the model to capture long-range dependencies in text
    - Multiple heads enable learning different aspects of relationships
    - Positional encoding preserves word order information
    - Layer normalization and residual connections enable stable training
    - Global average pooling provides a fixed-size representation regardless of input length
    """
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4, 
                 d_ff=512, max_len=256, num_classes=2, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Stack of Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Create padding mask
        if mask is None:
            mask = (x != 0).unsqueeze(1).unsqueeze(2)  # Shape: (batch, 1, 1, seq_len)
        
        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        # Pass through transformer encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
        
        # Global average pooling (considering padding)
        padding_mask = (mask.squeeze(1).squeeze(1)).float()  # Shape: (batch, seq_len)
        x = x * padding_mask.unsqueeze(-1)
        x = x.sum(dim=1) / padding_mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device='cpu'):
        """
        Load a model from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load the model on
            
        Returns:
            model: Loaded TransformerClassifier model
            vocab: Vocabulary dictionary
            config: Model configuration dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
        vocab = checkpoint['vocab']
        
        model = cls(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            max_len=config['max_len'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model, vocab, config


def get_model_summary(model):
    """
    Get a summary of the model architecture.
    
    Args:
        model: TransformerClassifier model
        
    Returns:
        str: Model summary string
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    TRANSFORMER CLASSIFIER ARCHITECTURE                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║   Input Token Indices                                                         ║
║         ↓                                                                     ║
║   ┌─────────────────┐                                                         ║
║   │   Embedding     │  (vocab_size → d_model)                                 ║
║   └────────┬────────┘                                                         ║
║            ↓                                                                  ║
║   ┌─────────────────┐                                                         ║
║   │   Positional    │  (Sinusoidal encoding)                                  ║
║   │   Encoding      │                                                         ║
║   └────────┬────────┘                                                         ║
║            ↓                                                                  ║
║   ┌─────────────────────────────────────────┐                                 ║
║   │         Transformer Encoder Block (x4)  │                                 ║
║   │   ┌─────────────────────────────────┐   │                                 ║
║   │   │   Multi-Head Self-Attention     │   │                                 ║
║   │   │   (8 heads, d_model=256)        │   │                                 ║
║   │   └──────────────┬──────────────────┘   │                                 ║
║   │                  ↓                      │                                 ║
║   │   ┌─────────────────────────────────┐   │                                 ║
║   │   │   Add & LayerNorm               │   │                                 ║
║   │   └──────────────┬──────────────────┘   │                                 ║
║   │                  ↓                      │                                 ║
║   │   ┌─────────────────────────────────┐   │                                 ║
║   │   │   Feed-Forward Network          │   │                                 ║
║   │   │   (d_model → d_ff → d_model)    │   │                                 ║
║   │   └──────────────┬──────────────────┘   │                                 ║
║   │                  ↓                      │                                 ║
║   │   ┌─────────────────────────────────┐   │                                 ║
║   │   │   Add & LayerNorm               │   │                                 ║
║   │   └─────────────────────────────────┘   │                                 ║
║   └─────────────────────────────────────────┘                                 ║
║            ↓                                                                  ║
║   ┌─────────────────┐                                                         ║
║   │  Global Average │  (Masked pooling)                                       ║
║   │    Pooling      │                                                         ║
║   └────────┬────────┘                                                         ║
║            ↓                                                                  ║
║   ┌─────────────────┐                                                         ║
║   │  Classification │  (d_model → d_model//2 → num_classes)                   ║
║   │      Head       │                                                         ║
║   └────────┬────────┘                                                         ║
║            ↓                                                                  ║
║   Output Logits (Positive/Negative)                                           ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}
"""
    return summary


if __name__ == "__main__":
    # Test the model
    print("Testing TransformerClassifier...")
    
    # Create a sample model
    model = TransformerClassifier(
        vocab_size=25000,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        max_len=256,
        num_classes=2,
        dropout=0.1
    )
    
    # Print model summary
    print(get_model_summary(model))
    
    # Test forward pass
    batch_size = 4
    seq_len = 256
    sample_input = torch.randint(0, 25000, (batch_size, seq_len))
    
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print("\nModel test passed!")
