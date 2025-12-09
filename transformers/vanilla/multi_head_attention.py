# multi_head_attention.py
#
import torch
import torch.nn as nn
import math
from vanilla import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads) -> None:
        """
        Multi-Head Attention layer.

        Args:
            d_model: Dimension of the model (e.g., 512)
            num_heads: Number of attention heads (e.g., 8)
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for Multi-Head Attention.

        Args:
            query, key, value: Tensors of shape (batch_size, seq_len, d_model)
            mask: Optional tensor for masking (e.g., causal mask in decoder)

        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
            attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into multiple heads: (batch_size, seq_len, num_heads, d_k) -> transpose
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(
            1, 2
        )  # (batch, h, seq, d_k)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply scaled dot-product attention
        if mask is not None:
            # Adjust mask shape for multi-head: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1)  # Add head dimension

        attention_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask
        )

        # Concatenate heads: (batch, h, seq, d_k) -> (batch, seq, h*d_k)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(attention_output)
        return output, attention_weights
