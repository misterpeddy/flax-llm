import fiddle as fdl
import jax.numpy as jnp

from models import bigram
import trainer
from data import data
from models import char_tokenizer


def get_cfg() -> fdl.Config[trainer.Trainer]:

    # TODO: convert to fiddle config.
    tokenizer = char_tokenizer.Tokenizer.from_text(char_tokenizer.EXAMPLE_INPUT_PATH)

    mdl_cfg = fdl.Config(
        bigram.BigramLM,
        vocab_size=tokenizer.vocab_size,
    )

    return fdl.Config(
        trainer.Trainer,
        model=mdl_cfg,
        num_steps=100,
        seed=42,
        batch_size=8,
        block_size=2,
        lr=1e-3,
        tokenizer=tokenizer,
        train_data_path=char_tokenizer.EXAMPLE_INPUT_PATH,
    )
