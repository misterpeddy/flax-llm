import sys

import fiddle as fdl
from flax import linen as nn
import jax.numpy as jnp

import trainer as train_lib
from models import char_tokenizer
from cfg import gpt2_117m

def get_cfg(name: str) -> fdl.Config[train_lib.Trainer]:
    return {
        "gpt2-117m": gpt2_117m.get_cfg,
    }[name]()

if __name__ == "__main__":
    assert len(sys.argv) < 3
    cfg_name = sys.argv[1] if len(sys.argv) == 2 else "gpt2-117m"
    trainer = fdl.build(get_cfg(cfg_name))
    trainer.train()
