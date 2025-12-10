# head_spliter.py
#
import torch


def head_splitter(input: torch.Tensor, num_heads: int, d_k: int) -> torch.Tensor:
    batch_size, seq_length, d_model = input.size()
    return input.view(batch_size, seq_length, num_heads, d_k).transpose(1, 2)
