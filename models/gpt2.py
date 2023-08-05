"""Implementation of GPT-2."""

from dataclasses import dataclass
from flax import linen as nn
import jax
import jax.numpy as jnp

from typing import Optional


class MultiHeadAttention(nn.Module):

    num_heads: int = 12     # H
    dtype: jnp.dtype = jnp.float32
    dropout_rate: float = 0.1
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array, deterministic: bool = None):
        B, T, C = x.shape
        if C % self.num_heads != 0:
            raise ValueError(
                f"Number of channels ({C}) not divisible by number of heads ({self.num_heads}).")
        head_dim = C // self.num_heads  # D

        # Define K, Q, V matrices.
        # [B, T, 3 * C]
        qkv = nn.Dense(3 * C, name="qkv", use_bias=self.use_bias)(x)
        qkv = jnp.reshape(qkv, (B, T, 3 * self.num_heads,
                          head_dim))  # [B, T, 3 * H, D]
        k, q, v = jnp.split(qkv, 3, axis=2)   # [B, T, H, D]

        # Calculate self attention matrix.
        attn = jnp.einsum("bthd,bThd->bhtT", q, k) / \
            jnp.sqrt(head_dim).astype(self.dtype)
        attn = jnp.where(mask, attn, jnp.finfo(attn.dtype).min)
        attn = jax.nn.softmax(attn, axis=-1).astype(self.dtype)
        attn = nn.Dropout(self.dropout_rate)(
            attn, deterministic)   # [B, H, T, T]

        # Calculate value weighted sum for each query position.
        x = jnp.einsum("bhtT,bThd->bthd", attn, v)  # [B, T, H, D]
        x = jnp.reshape(x, (B, T, C))   # [B, T, C]
        x = nn.Dense(features=C, name="out_proj", dtype=self.dtype,
                     use_bias=self.use_bias)(x)  # [B, T, C]
        x = nn.Dropout(self.dropout_rate)(x, deterministic)

        return x


class MLP(nn.Module):

    dtype: jnp.dtype = jnp.float32
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool | None = None) -> jax.Array:
        _, _, C = x.shape
        x = nn.Dense(features=4 * C, dtype=self.dtype, name="c_fc")(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(features=C, dtype=self.dtype, name="c_proj")(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic)
        return x


class Block(nn.Module):

    ln_epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32
    use_bias: bool = True
    num_heads: int = 12
    dropout_rate: float = 0.1

    def setup(self):
        self.pre_attn_ln = nn.LayerNorm(epsilon=self.ln_epsilon, dtype=self.dtype,
                                        use_bias=self.use_bias)
        self.attn = MultiHeadAttention(
            self.num_heads, self.dtype, dropout_rate=self.dropout_rate)
        self.post_attn_ln = nn.LayerNorm(
            epsilon=self.ln_epsilon, dtype=self.dtype, use_bias=self.use_bias)
        self.mlp = MLP()

    def __call__(self, x, mask=None, deterministic=None) -> jax.Array:
        x = x + self.attn(self.pre_attn_ln(x), mask, deterministic)
        mlp = self.mlp(self.post_attn_ln(x), deterministic)
        x = x + mlp
        return x


class GPT2LM(nn.Module):

    vocab_size: int = 50304
    num_embeds: int = 768
    block_size: int = 1024
    num_layers: int = 12
    dtype: jnp.dtype = jnp.float32
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jax.Array, deterministic=True):
        B, T = x.shape

        pos = jnp.expand_dims(jnp.arange(0, T), 0)   # [1, T]
        attn_mask = nn.make_causal_mask(x, dtype=bool)  # [B, 1, T, T]

        wte = nn.Embed(num_embeddings=self.vocab_size,
                       features=self.num_embeds, dtype=self.dtype, name="wte")
        wpe = nn.Embed(num_embeddings=self.block_size,
                       features=self.num_embeds, dtype=self.dtype, name="wpe")

        token_embed = wte(x)    # [B, T, C]
        pos_embed = wpe(pos)    # [1, T, C]
        combined_embed = token_embed + pos_embed
        x = nn.Dropout(rate=self.dropout_rate)(
            combined_embed, deterministic)

        for i in range(self.num_layers):
            x = Block(name=f"block_{i}")(
                x, attn_mask, deterministic)

        x = nn.LayerNorm(name="layer_norm")(x)
        logits = wte.attend(x)
        return logits


if __name__ == "__main__":
    """Example usage."""
    model = GPT2LM()
    rng_key = jax.random.PRNGKey(seed=42)
    batch_size = 4
    block_size = 8
    random_x = jax.random.randint(key=rng_key, shape=(
        batch_size, block_size), dtype=int, minval=0, maxval=model.vocab_size)

    params = model.init(rng_key, random_x)

    gen = model.apply(params, x=random_x)

    print(gen)
