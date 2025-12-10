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
            dropout: Dropout rate (default: 0.1)
        """
        super(DecoderLayer, self).__init__()
        # Sublayer 1: Masked Self-Attention (decoder's own previous outputs)
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # Sublayer 2: Encoder-Decoder Attention (queries from decoder, keys/values from encoder)
        self.encoder_attn = MultiHeadAttention(d_model, num_heads)

        # Sublayer 3: Position-wise Feed-Forward Network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        # Layer Normalization (one for each sublayer)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
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
        # ---------- Sublayer 1: Masked Self-Attention ----------
        # Query, Key, Value all come from decoder input (x)
        self_attn_output, _ = self.self_attn(query=x, key=x, value=x, mask=tgt_mask)
        self_attn_output = self.dropout(self_attn_output)
        x = self.norm1(x + self_attn_output)  # Residual + LayerNorm

        # ---------- Sublayer 2: Encoder-Decoder Attention ----------
        # Query: from decoder (x)
        # Key, Value: from encoder (encoder_output)
        enc_attn_output, _ = self.encoder_attn(
            query=x, key=encoder_output, value=encoder_output, mask=src_mask
        )
        enc_attn_output = self.dropout(enc_attn_output)
        x = self.norm2(x + enc_attn_output)  # Residual + LayerNorm

        # ---------- Sublayer 3: Feed-Forward Network ----------
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm3(x + ff_output)  # Residual + LayerNorm

        return x

