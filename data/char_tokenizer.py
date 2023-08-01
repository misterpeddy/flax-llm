from typing import Any, Callable, Tuple

import jax
from jax import numpy as jnp

import os
print(os.getcwd())
_INPUT_PATH = "./data/shakespeare.txt"

def get_enc_dec() -> Tuple[Callable[[str], list[int]], Callable[[Any], str]]:
  with open(_INPUT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

  chars = sorted(list(set(text)))
  vocab_size = len(chars)
  
  stoi = { ch:i for i,ch in enumerate(chars) }
  itos = { i:ch for i,ch in enumerate(chars) }

  def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]
  def decode(l: list[int] | jax.Array) -> str:
    if isinstance(l, jax.Array):
      if l.ndim == 0:
        l = jnp.expand_dims(l, 0)
      l = jax.device_get(l)
    return "".join([itos[i] for i in l])
  return encode, decode

encode, decode = get_enc_dec()
print(decode(encode("hello world")))