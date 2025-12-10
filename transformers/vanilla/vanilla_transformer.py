# Vanilla Transformer
#
import torch
import torch.nn as nn
from typing import Tuple
import math


class VanillaTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout=0.1,
        max_seq_len=5000,
    ) -> None:
        """
        Full Vanilla Transformer model as described in "Attention Is All You Need".

        Args:
            src_vocab_size: Size of source vocabulary (e.g., 10000 for English)
            tgt_vocab_size: Size of target vocabulary (e.g., 10000 for French)
            d_model: Dimension of the model (default: 512)
            num_heads: Number of attention heads (default: 8)
            d_ff: Hidden dimension of feed-forward layers (default: 2048)
            num_encoder_layers: Number of encoder layers (N, default: 6)
            num_decoder_layers: Number of decoder layers (N, default: 6)
            dropout: Dropout rate (default: 0.1)
            max_seq_len: Maximum sequence length for positional encoding (default: 5000)
        """

        super(VanillaTransformer, self).__init__()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass of the Vanilla Transformer.

        Args:
            src: Source input tensor (batch_size, src_seq_len)
            tgt: Target input tensor (batch_size, tgt_seq_len)
            src_mask: Optional padding mask for source (batch_size, 1, src_seq_len)
            tgt_mask: Optional padding mask for target (batch_size, 1, tgt_seq_len)

        Returns:
            Output tensor (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        pass  # Implementation of forward pass goes here
