# position_wise_feed_forward_network.py
#

from typing import Optional
from flax import nnx
import jax.numpy as jnp


class PositionWiseFeedForward(nnx.Module):
    def __init__(
        self, d_model: int, d_ff: int, rngs: nnx.Rngs, dropout: float = 0.01
    ) -> None:
        self.linear_in = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.linear_out = nnx.Linear(d_ff, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(dropout)

    def __call__(self, x, rngs: Optional[nnx.Rngs] = None) -> jnp.ndarray:
        x = nnx.relu(self.linear_in(x))
        x = self.dropout(x, rngs=rngs)
        x = self.linear_out(x)
        return x
