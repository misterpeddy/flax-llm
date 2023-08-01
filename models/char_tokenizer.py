from typing import Any, Callable, Tuple

import jax
from jax import numpy as jnp

EXAMPLE_INPUT_PATH = "./data/shakespeare.txt"


class Tokenizer:
    """Simple tokenizer for character-level language models."""

    def __init__(self, chars: list[str]):
        self.vocab_size = len(chars)
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
    
    @classmethod
    def from_text(cls, file_path: str) -> "Tokenizer":
        """Creates a tokenizer from a text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(list(set(text))) 
        return cls(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, l: list[int] | jax.Array) -> str:
        if isinstance(l, jax.Array):
            if l.ndim == 0:
                l = jnp.expand_dims(l, 0)
            l = jax.device_get(l)
        return "".join([self.itos[i] for i in l])


if __name__ == "__main__":
    tokenizer = Tokenizer.from_text(EXAMPLE_INPUT_PATH)
    print(tokenizer.decode(tokenizer.encode("hello world")))
