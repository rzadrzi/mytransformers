# MultiHeadAttention Class
#

from flax import nnx
import jax.numpy as jnp
import math


def scaled_dot_production_attention(query, key, value, mask=None):
    d_k = query.shape[-1]
    scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = jnp.where(mask == 0, -1e16, scores)

    attention_weights = nnx.softmax(scores, axis=-1)
    output = jnp.matmul(attention_weights, value)
    return output, attention_weights


class MultiHeadAttentionJAX(nnx.Module):
    def __init__(self):
        pass


"""

if __name__ == "__main__":
    from jax import random

    rngs = random.PRNGKey(42)
    seq_len, d_k = 3, 2
    rngs, rand1 = random.split(rngs)
    qkv = random.normal(rand1, (3, seq_len, d_k))
    q, k, v = qkv[0], qkv[1], qkv[2]
    values, attention = scaled_dot_production_attention(q, k, v)
    print("Q\n", q)
    print("K\n", k)
    print("V\n", v)
    print("Values\n", values)
    print("Attention\n", attention)

"""
