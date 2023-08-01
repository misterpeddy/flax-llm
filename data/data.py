from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable
import jax
from jax import numpy as jnp


@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, s: str) -> list[int]:
        ...

    def decode(self, l: list[int] | jax.Array) -> str:
        ...

@dataclass
class Dataset:
    """Simple dataset iterator class."""
    data: jax.Array
    batch_size: int
    block_size: int
    loop: bool = field(default=True)

    def __post_init__(self):
        if self.data.ndim != 1:
            raise ValueError(f"Expected data to be 1D, got {self.data.ndim}D.")
        self.num_batches = self.data.shape[0] // (
            self.block_size * self.batch_size)
        self._batch_idx = 0

    @classmethod
    def from_file(cls, path: str, tokenizer: Tokenizer, **kwargs) -> "Dataset":
        """Creates a dataset from a text file."""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        data = jnp.array(tokenizer.encode(text))
        return cls(data, **kwargs)

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> tuple[jax.Array, jax.Array]:
        if self._batch_idx >= self.num_batches:
            if self.loop:
                self._batch_idx = 0
            else:
                raise StopIteration
        start_idx = self._batch_idx * self.batch_size * self.block_size
        end_idx = (self._batch_idx + 1) * self.batch_size * self.block_size
        x = self.data[start_idx : end_idx].reshape(self.batch_size, self.block_size)
        y = self.data[start_idx + 1 : end_idx + 1].reshape(self.batch_size, self.block_size)
        self._batch_idx += 1
        return x, y
