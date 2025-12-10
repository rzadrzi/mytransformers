# multi_head_attention.py
#
import torch
import torch.nn as nn
from vanilla import scaled_dot_product_attention, head_splitter, heads_combiner
from typing import Tuple


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,  # Dimensionality of the input.
        num_heads: int,  # The number of attention heads to split the input into.
    ) -> None:
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

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Multi-Head Attention.

        Args:
            query, key, value: Tensors of shape (batch_size, seq_len, d_model)
            mask: Optional tensor for masking (e.g., causal mask in decoder)

        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
            attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
        """
        #        batch_size = query.size(0)

        Q = head_splitter(self.W_q(query), self.num_heads, self.d_k)
        K = head_splitter(self.W_k(key), self.num_heads, self.d_k)
        V = head_splitter(self.W_v(value), self.num_heads, self.d_k)

        # Apply scaled dot-product attention
        if mask is not None:
            # Adjust mask shape for multi-head: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1)  # Add head dimension

        attention_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask
        )

        # Final linear projection
        output: torch.Tensor = self.W_o(heads_combiner(attention_output, self.d_model))
        return output, attention_weights
