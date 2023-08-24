from typing import Mapping, Tuple, Optional, Callable
import jax
import jax.numpy as jnp
import flax.core
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from jaxlm.utils.jax import (
    JaxRNG, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, get_weight_decay_mask,
    make_shard_and_gather_fns, with_sharding_constraint,
)
from functools import partial

INPUT_TOKEN_KEY = "input_ids"
LABELS_KEY = "labels"
LOSS_MASK_KEY = "loss_masks"  # NOTE: loss is only calculated for loss_masks == 1

def loss_and_accuracy(params, batch, rng, model=None):
    batch = with_sharding_constraint(
        batch, PS(('dp', 'fsdp'))
    )
    logits = model.apply(
        params, batch[INPUT_TOKEN_KEY], deterministic=False,
        rngs=rng,
    ).logits
    loss, accuracy = cross_entropy_loss_and_accuracy(
        logits, batch[LABELS_KEY], batch[LOSS_MASK_KEY]
    )
    return loss, {'accuracy': accuracy}


def get_microbatch(batch: dict, idx: int, microbatch_size: int) -> Mapping[str, jnp.ndarray]:
    """Fetch microbatch slice from possibly-packed input data."""
    offset = idx * microbatch_size
    length = microbatch_size
    starts = {k: [offset] + [0] * (b.ndim - 1)
              for k, b in batch.items()}
    limits = {k: [length] + list(b.shape[1:])
              for k, b in batch.items()}
    return {
        k: jax.lax.dynamic_slice(b, starts[k], limits[k])
        for k, b in batch.items()
    }


def train_step_microbatched(
    train_state, batch,
    rng, rng_keys,
    model=None,
    num_microbatches: int = 1,
    accum_dtype=jnp.float32,
    learning_rate_schedule: Callable[[int], float] = None,
):
    """Implements optional microbatched gradient accumulation.

    Args:
    loss_fn: The loss function that takes in (train_state.params, batch, dropout_rng).
    train_state: A train state with model parameters and optimizer state.
    batch: A batch of data.
    dropout_rng: jax PRNGKey for dropout.
    num_microbatches: the number of microbatches to use, or None for direct
        training.
    data_partition_spec: the PartitionSpec to use for partitioning annotations
        on the batch.

    Returns:
    Accumulated gradients and incremental metrics.
    """
    batch_size = batch[INPUT_TOKEN_KEY].shape[0]
    microbatch_size = batch_size // num_microbatches
    assert batch_size % num_microbatches == 0, (
        f'Batch size ({batch_size}) must be divisible by the number of '
        f'microbatches ({num_microbatches}).'
    )

    loss_and_accuracy_fn = partial(loss_and_accuracy, model=model)
    get_microbatch_fn = partial(
        get_microbatch, microbatch_size=microbatch_size)
    grad_fn = jax.value_and_grad(loss_and_accuracy_fn, has_aux=True)

    def calculate_grad(loop_cnt, rng):
        mbatch = get_microbatch_fn(batch, loop_cnt)
        # We need to annotate the microbatch sharding as we would do to a batch.
        mbatch = jax.tree_util.tree_map(
            lambda x: with_sharding_constraint(x, PS('dp')),
            mbatch
        )
        (loss, metrics), grad = grad_fn(
            train_state.params,
            mbatch,
            rng,
        )
        return loss, grad, metrics

    def per_microbatch_train_step(
        loop_cnt: int,
        state: Tuple[jnp.ndarray, jnp.ndarray,
                     Mapping[str, jnp.ndarray],
                     Optional[flax.core.FrozenDict]]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Mapping[str, jnp.ndarray],
               Optional[flax.core.FrozenDict]]:
        (rng, loss_accum, grad_accum, metrics_accum) = state
        loss, grad, metrics = calculate_grad(loop_cnt, rng)

        # convert to accum_dtype
        loss = loss.astype(accum_dtype)
        grad = jax.tree_util.tree_map(
            lambda x: x.astype(accum_dtype), grad
        )
        metrics = jax.tree_util.tree_map(
            lambda x: x.astype(accum_dtype), metrics
        )

        loss_accum = loss_accum + loss
        metrics_accum = jax.tree_util.tree_map(
            jnp.add, metrics_accum, metrics
        )
        grad_accum = jax.tree_util.tree_map(jnp.add, grad_accum, grad)
        return rng, loss_accum, grad_accum, metrics_accum

    # Initialize gradient accumulation loop state.
    loss_accum_init = jnp.zeros((), accum_dtype)
    grad_accum_init = jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape, accum_dtype),
        train_state.params
    )

    rng_generator = JaxRNG(rng)
    input_rng = rng_generator(rng_keys)
    _, _, initial_metrics_shape = jax.eval_shape(
        calculate_grad, loop_cnt=0,
        rng=input_rng
    )

    metrics_accum_init = {
        k: jnp.zeros((), accum_dtype)
        for k in initial_metrics_shape
    }
    loop_init = (
        input_rng,  # same rng for all microbatches
        loss_accum_init,
        grad_accum_init,
        metrics_accum_init
    )
    _, loss_accum, grad_accum, metrics_accum = jax.lax.fori_loop(
        0, num_microbatches, per_microbatch_train_step, loop_init
    )

    # Apply the gradients to the model.
    train_state = train_state.apply_gradients(grads=grad_accum)
    metrics = dict(
        loss=loss_accum / num_microbatches,
        accuracy=metrics_accum['accuracy'] / num_microbatches,
        learning_rate=learning_rate_schedule(train_state.step),
        gradient_norm=global_norm(grad_accum),
        param_norm=global_norm(train_state.params),
    )
    new_rng = rng_generator()
    return train_state, new_rng, metrics


def eval_step(
    train_state, batch, rng, rng_keys, model=None,
):
    rng_generator = JaxRNG(rng)
    batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
    logits = model.apply(
        train_state.params, batch[INPUT_TOKEN_KEY], deterministic=True,
        rngs=rng_generator(rng_keys),
    ).logits
    loss, accuracy = cross_entropy_loss_and_accuracy(
        logits, batch[LABELS_KEY], batch[LOSS_MASK_KEY]
    )
    metrics = dict(
        eval_loss=loss,
        eval_accuracy=accuracy,
    )
    return rng_generator(), metrics
