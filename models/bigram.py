# Simple bigram language model

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp

FrozenDict = flax.core.FrozenDict


class BigramLM(nn.Module):

    vocab_size: int

    @nn.compact
    def __call__(self, inputs):
        """Looks up inputs in a lookup table to assign log prob to each vocab element."""
        return nn.Embed(num_embeddings=self.vocab_size, features=self.vocab_size)(inputs)

    def generate(self, rng_key: jax.Array, params: FrozenDict, inputs: jax.Array, seq_len: int = 1) -> jax.Array:
        """Continues inputs [B, T] to generate [B, seq_len] outputs."""
        inputs = inputs[:, -1]
        gen = jnp.zeros((inputs.shape[0], seq_len))
        for i in range(seq_len):
            logits = self.apply(variables=params, inputs=inputs)
            rng_key, rng_subkey = jax.random.split(rng_key)
            top_1 = jax.random.categorical(rng_subkey, logits)
            gen = gen.at[:, i].set(top_1)
            inputs = top_1
        return gen


if __name__ == "__main__":
    """Example usage."""
    vocab_size = 256
    model = BigramLM(vocab_size=vocab_size)

    random_key = jax.random.PRNGKey(seed=42)
    batch_size = 4
    block_size = 8
    random_x = jax.random.randint(key=random_key, shape=(
        batch_size, block_size), dtype=int, minval=0, maxval=vocab_size)

    params = model.init(random_key, random_x)

    gen = model.generate(random_key, params, inputs=random_x, seq_len=2)

    print(gen)
