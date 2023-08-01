from models import char_tokenizer
from models import utils as model_utils
from data import data

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

def get_model(model_name: str) -> nn.Module:
    if model_name == "gpt2":
        from models import gpt2
        return gpt2.GPT2LM
    elif model_name == "bigram":
        from models import bigram
        return bigram.BigramLM
    else:
        raise ValueError(f"Unknown model {model_name}.")

def train(
        model_name: str = "gpt2",
        num_steps: int = 100,
        seed: int = 42,
        batch_size: int = 64,
        block_size: int = 2,
        lr: int = 1e-3) -> tuple[nn.Module, jnp.ndarray]:
    """Trains a bigram language model on a toy dataset."""
    rng_key = jax.random.PRNGKey(seed=seed)

    # Initialize model.
    tokenizer = char_tokenizer.Tokenizer.from_text(
        char_tokenizer.EXAMPLE_INPUT_PATH)
    model = get_model(model_name)
    params = model.init(rng_key, jnp.zeros(
        (batch_size, block_size), dtype=jnp.int32))

    # Initialize optimizer.
    optimizer = optax.adam(learning_rate=lr)
    optimizer_state = optimizer.init(params)

    # Initialize dataset.
    dataset = data.Dataset.from_file(char_tokenizer.EXAMPLE_INPUT_PATH,
                                     tokenizer=tokenizer, batch_size=batch_size, block_size=block_size)

    # Run training loop.
    for i in range(num_steps):
        x, y = next(dataset)
        loss, grads = jax.value_and_grad(
            model_utils.cross_entropy_loss(model))(params, x, y)
        print(f"Step {i} | Loss: {loss:.3f}")
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)

    return model, params


if __name__ == "__main__":
    train()
