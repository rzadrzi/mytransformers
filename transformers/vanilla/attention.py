import torch
import math


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Computes scaled dot-product attention.

    Args:
        query, key, value: Tensors of shape (batch_size, num_heads, seq_len, d_k)
        mask: Optional tensor of shape (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)

    Returns:
        output: Tensor of shape (batch_size, num_heads, seq_len, d_k)
        attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
        d_k
    )  # (..., seq_len, seq_len)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights
