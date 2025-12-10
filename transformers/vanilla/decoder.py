# decoder.py
#
import torch.nn as nn
from vanilla import MultiHeadAttention, PositionWiseFeedForward 

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        One layer of the Transformer Decoder.
        Args:
            d_model: Dimension of the model (e.g., 512)
            num_heads: Number of attention heads (e.g., 8)
            d_ff: Dimension of the feed-forward hidden layer (e.g., 2048)
            dropout: Dropout rate for regularization (default: 0.1)
        """
        super(DecoderLayer, self).__init__()

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Forward pass of the Decoder layer.
        Args:
            x: Input tensor of shape (batch_size, tgt_seq_len, d_model)
            enc_output: Encoder output tensor of shape (batch_size, src_seq_len, d_model)
            src_mask: Optional mask for source padding (shape: batch_size, 1, src_seq_len)
                      Used to prevent attention to padding tokens in the encoder output.
            tgt_mask: Optional mask for target (shape: batch_size, 1, tgt_seq_len)
                      Used to prevent attention to future tokens in the decoder input.
        Returns:
            Output tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        pass # Implementation of forward pass goes here