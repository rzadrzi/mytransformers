# positional encoding
# non-learnable
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
        # If you have parameters in your model, which should be saved and restored in the state_dict,
        # but not trained by the optimizer, you should register them as buffers.
        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Tensor with positional encodings added, same shape as input.
        """
        # x shape: (B, L, D)
        # pe shape: (L, D) → we slice up to L and add (broadcasting)
        return x + self.pe[: x.size(1)]


if __name__ == "__main__":
    d_model = 10
    max_len = 10
    seq_len = 5
    batch_size = 2

    pe = PositionalEncoding(d_model, max_len)
    x = torch.zeros(batch_size, seq_len, d_model)
    test = pe(x)

    print(pe.pe)
    #    print(x.size())
    #    print(x)
    #    print(test)

    print(pe.pe[: x.size(1)])
