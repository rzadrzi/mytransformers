# _heads_combiner.py
#

import torch


def heads_combiner(input: torch.Tensor, d_model: int) -> torch.Tensor:
    batch_size, _, seq_length, d_k = input.size()
    return input.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)
