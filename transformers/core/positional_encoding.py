# positional encoding
#
#

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        """
        Initialize Positional Encoding Module

        Args:
            d_model (int): Dimensionality of the model/embedding.
            max_len (int): Maximum sequence length to pre-compute encodings for.

        """
        super(PositionalEncoding, self).__init__()
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Create position indices: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the div_term: 1 / (10000^(2i / d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions: cos

        # Register as a buffer (not a model parameter, but part of state)
        self.register_buffer("pe", pe)


p = PositionalEncoding(10, 10)
print(p.pe)
# print(p.position)
# print(p.div_term)
