from flax import linen as nn
import jax.numpy as jnp

import trainer as train_lib
from models import char_tokenizer


def get_model(model_name: str) -> nn.Module:
    if model_name == "gpt2":
        from models import gpt2
        return gpt2.GPT2LM
    elif model_name == "bigram":
        from models import bigram
        return bigram.BigramLM
    else:
        raise ValueError(f"Unknown model {model_name}.")

def _get_config(model_name: str = "gpt2") -> train_lib.Trainer:
    model_cls = get_model(model_name)
    return train_lib.Trainer(
        model_cls=model_cls, 
        num_steps= 100,
        seed= 42,
        batch_size = 64,
        block_size = 2,
        lr= 1e-3,
        tokenizer_cls=char_tokenizer.Tokenizer,
        train_data_path=char_tokenizer.EXAMPLE_INPUT_PATH,
    )

def train() -> tuple[nn.Module, jnp.ndarray]:
    """Trains a language model according to provided config."""
    trainer = _get_config()
    model = trainer.train()
    del model, params

if __name__ == "__main__":
    train()
