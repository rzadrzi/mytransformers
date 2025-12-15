# positional_encoding.py
#

import jax
import jax.numpy as jnp
from flax import nnx


class PositionalEncoding(nnx.Module):
    def __init__(
        self, d_model: int, max_len: int, base=1000.0, dtype=jnp.float32
    ) -> None:
        assert d_model % 2 == 0, "d_model must be even in PositionalEncoding"
        pos = jnp.arange(max_len, dtype=dtype)[:, None]
        i = jnp.arange(0, d_model, 2, dtype=dtype)[None, :]

        inv_freq = 1.0 / (base ** (i / d_model))

        angle = pos * inv_freq

        pe = jnp.zeros((max_len, d_model), dtype=dtype)
        pe = pe.at[:, 0::2].set(jnp.sin(angle))
        pe = pe.at[:, 1::2].set(jnp.cos(angle))
        self.pe = jax.device_put(pe)

    def __call__(self, x):
        x = x + self.pe[:, : x.shape[1]]
        return x


"""
def positional_encoding(
    d_model: int, max_len: int, base=1000.0, dtype=jnp.float32
) -> jnp.ndarray:
    assert d_model % 2 == 0, "d_model must be even for positional_encoding"
    pos = jnp.arange(max_len, dtype=dtype)[:, None]

    i = jnp.arange(0, d_model, 2, dtype=dtype)[None, :]

    inv_freq = 1.0 / (base ** (i / d_model))

    angle = pos * inv_freq

    pe = jnp.zeros((max_len, d_model), dtype=dtype)
    pe = pe.at[:, 0::2].set(jnp.sin(angle))
    pe = pe.at[:, 1::2].set(jnp.cos(angle))
    return pe

"""
