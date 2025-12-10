# encoder.py
#
#
import torch.nn as nn
from vanilla import MultiHeadAttention, PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        One layer of the Transformer Encoder.
        Args:
            d_model: Dimension of the model (e.g., 512)
            num_heads: Number of attention heads (e.g., 8)
            d_ff: Dimension of the feed-forward hidden layer (e.g., 2048)
            dropout: Dropout rate for regularization (default: 0.1)
        """
        super(EncoderLayer, self).__init__()

        # Sublayer 1: Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # Sublayer 2: Position-wise Feed-Forward Network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        # Layer Normalization (one for each sublayer)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        Forward pass of the Encoder layer.
        Args:
            x: Input tensor of shape (batch_size, src_seq_len, d_model)
            src_mask: Optional mask for padding (shape: batch_size, 1, src_seq_len)
                      Used to prevent attention to padding tokens.
        Returns:
            Output tensor of shape (batch_size, src_seq_len, d_model)
        """
        # ---------- Sublayer 1: Self-Attention ----------
        # Compute attention output
        attn_output, _ = self.self_attn(query=x, key=x, value=x, mask=src_mask)
        # Apply dropout to attention output
        attn_output = self.dropout(attn_output)
        # Add residual connection and apply layer normalization
        x = self.norm1(x + attn_output)  # Shape: (batch, seq_len, d_model)

        # ---------- Sublayer 2: Feed-Forward Network ----------
        # Compute FFN output
        ff_output = self.feed_forward(x)
        # Apply dropout to FFN output
        ff_output = self.dropout(ff_output)
        # Add residual connection and apply layer normalization
        x = self.norm2(x + ff_output)  # Shape: (batch, seq_len, d_model)

        return x
