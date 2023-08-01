import jax
import flax
from flax import linen as nn
import optax

FrozenDict = flax.core.FrozenDict

def cross_entropy_loss(model: nn.Module):
    """Computes cross entropy loss on model generation."""
    @jax.jit
    def _cross_entropy_loss(params: FrozenDict, inputs: jax.Array, targets: jax.Array):
        logits = model.apply(params, inputs)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])
        return optax.softmax_cross_entropy(logits, one_hot_targets).mean()
    
    return _cross_entropy_loss

