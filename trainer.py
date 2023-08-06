from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from data import data
from models import utils as model_utils 


@dataclass
class Trainer:
    model: Any
    num_steps: int
    seed: int
    batch_size: int
    block_size: int
    lr: float

    train_data_path: str

    tokenizer: data.Tokenizer
    loss_fn: Callable[[nn.Module], Callable] = model_utils.cross_entropy_loss

    def train(self):
        rng_key = jax.random.PRNGKey(seed=self.seed)

        # Initialize model.
        params = self.model.init(rng_key, jnp.zeros(
            (self.batch_size, self.block_size), dtype=jnp.int32))

        # Initialize optimizer.
        optimizer = optax.adam(learning_rate=self.lr)
        optimizer_state = optimizer.init(params)

        # Initialize dataset.
        dataset = data.Dataset.from_file(self.train_data_path,
                                         tokenizer=self.tokenizer,
                                         batch_size=self.batch_size,
                                         block_size=self.block_size)

        # Run training loop.
        for i in range(self.num_steps):
            x, y = next(dataset)
            loss_fn = self.loss_fn(self.model)
            loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
            print(f"Step {i} | Loss: {loss:.3f}")
            updates, optimizer_state = optimizer.update(grads, optimizer_state)
            params = optax.apply_updates(params, updates)

        return self.model, params
