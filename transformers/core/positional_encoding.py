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
            max_len (int):

        """
        super(PositionalEncoding, self).__init__()
        # Create a matrix of shape (max_len, d_model)
        self.pe = torch.zeros(max_len, d_model)

        # Create position indices: (max_len, 1)
        self.position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)


p = PositionalEncoding(10, 10)
print(p.pe)
print(p.position)
