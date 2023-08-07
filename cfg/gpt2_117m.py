import fiddle as fdl
import jax.numpy as jnp

from models import gpt2
import trainer
from models import char_tokenizer


def get_cfg() -> fdl.Config[trainer.Trainer]:

    # TODO: convert to fiddle config.
    tokenizer = char_tokenizer.Tokenizer.from_text(char_tokenizer.EXAMPLE_INPUT_PATH)

    mdl_cfg = fdl.Config(
        gpt2.GPT2LM,
        vocab_size=tokenizer.vocab_size,
        emb_dim=768,
        block_size=1024,
        num_layers=12,
        dtype=jnp.float32,
        dropout_rate=0.1
    )

    return fdl.Config(
        trainer.Trainer,
        model=mdl_cfg,
        num_steps=100,
        seed=42,
        batch_size=8,
        block_size=1024,
        lr=1e-3,
        tokenizer=char_tokenizer.Tokenizer,
        train_data_path=char_tokenizer.EXAMPLE_INPUT_PATH,
    )
