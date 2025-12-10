# position wise FFN

import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout=0.1) -> None:
        """
        Position-wise Feed-Forward Network as described in "Attention Is All You Need".
        Architecture:
            Input (d_model)
                → Linear → ReLU → Dropout → Linear
                → Output (d_model)
        Args:
            d_model: Dimension of the model (e.g., 512)
            d_ff: Dimension of the hidden layer (e.g., 2048)
            dropout: Dropout rate (default: 0.1)
        """

        super(PositionWiseFeedForward, self).__init__()
        self.linear_in = nn.Linear(d_model, d_ff)
        self.linear_out = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """

        x = self.linear_in(x)  # (batch, seq_len, d_ff)
        x = self.activation(x)  # (batch, seq_len, d_ff)
        x = self.dropout(x)  # (batch, seq_len, d_ff)
        return self.linear_out(x)  # (batch, seq_len, d_model)
