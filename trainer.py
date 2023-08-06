from dataclasses import dataclass
from typing import Any, Callable

import fiddle as fdl
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from data import data
from models import utils as model_utils 


@dataclass
class Trainer:
    model_cls: type[Any]
    num_steps: int
    seed: int
    batch_size: int
    block_size: int
    lr: float

    train_data_path: str

    tokenizer_cls: type[data.Tokenizer]
    loss_fn: Callable[[nn.Module], Callable] = model_utils.cross_entropy_loss

    def train(self):
        rng_key = jax.random.PRNGKey(seed=self.seed)

        # Initialize model.
        tokenizer = self.tokenizer_cls.from_text(self.train_data_path)
        model = self.model_cls(vocab_size=tokenizer.vocab_size)
        params = model.init(rng_key, jnp.zeros(
            (self.batch_size, self.block_size), dtype=jnp.int32))

        # Initialize optimizer.
        optimizer = optax.adam(learning_rate=self.lr)
        optimizer_state = optimizer.init(params)

        # Initialize dataset.
        dataset = data.Dataset.from_file(self.train_data_path,
                                         tokenizer=tokenizer,
                                         batch_size=self.batch_size,
                                         block_size=self.block_size)

        # Run training loop.
        for i in range(self.num_steps):
            x, y = next(dataset)
            loss_fn = self.loss_fn(model)
            loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
            print(f"Step {i} | Loss: {loss:.3f}")
            updates, optimizer_state = optimizer.update(grads, optimizer_state)
            params = optax.apply_updates(params, updates)

        return model, params