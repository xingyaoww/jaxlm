from typing import Mapping, Tuple, Optional, Callable
import jax
import jax.numpy as jnp
import flax.core
from flax.training.common_utils import onehot
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from jaxlm.utils.jax import (
    JaxRNG,
    global_norm,
    with_sharding_constraint,
)
from functools import partial

INPUT_TOKEN_KEY = "input_ids"
LABELS_KEY = "labels"
LOSS_WEIGHT_KEY = "loss_weight"  # =0 are masked out


def cross_entropy_loss(logits, labels, loss_weight=None):
    # shift logits and labels
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(
            jax.nn.log_softmax(shift_logits, axis=-1),
            jnp.expand_dims(shift_labels, -1),
            axis=-1,
        ),
        -1,
    )

    # We do not avg for each batch due to minibatch implementation
    loss = -jnp.sum(loss_weight * token_log_prob)

    valid_mask = loss_weight != 0.0
    n_correct = jnp.sum(
        jnp.where(
            valid_mask,
            jnp.argmax(shift_logits, axis=-1) == shift_labels,
            jnp.array(False)
        )
    )
    return loss, {
        "n_correct": n_correct,
        "n_tokens": jnp.sum(valid_mask),
    }


def loss_and_metrics(params, batch, rng_output, model=None):
    batch = with_sharding_constraint(
        batch, PS(('dp', 'fsdp'))
    )
    logits = model.apply(
        params, batch[INPUT_TOKEN_KEY], deterministic=False,
        rngs=rng_output,
    ).logits
    return cross_entropy_loss(
        logits, batch[LABELS_KEY], batch[LOSS_WEIGHT_KEY]
    )


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


def train_step(
    train_state,
    batch,
    rng_output,
    model=None,
    num_microbatches: int = 1,
    accum_dtype=jnp.float32,
    learning_rate_schedule: Callable[[int], float] = None,
):
    """Train model with (optional) microbatched gradient accumulation.

    Args:
    train_state: A train state with model parameters and optimizer state.
    batch: A batch of data.
    rng_output: jax PRNGKey output for model (e.g., for dropout).
        Instead of RNG, this make sure that the same random number is used for all microbatches.
        That is, the setting of num_microbatches should have no effect on the training result.
    num_microbatches: the number of microbatches to use, or None for direct
        training.
    """
    batch_size = batch[INPUT_TOKEN_KEY].shape[0]
    microbatch_size = batch_size // num_microbatches
    assert batch_size % num_microbatches == 0, (
        f'Batch size ({batch_size}) must be divisible by the number of '
        f'microbatches ({num_microbatches}).'
    )

    loss_and_metrics_fn = partial(loss_and_metrics, model=model)
    get_microbatch_fn = partial(
        get_microbatch, microbatch_size=microbatch_size)
    grad_fn = jax.value_and_grad(loss_and_metrics_fn, has_aux=True)

    def calculate_grad(loop_cnt):
        mbatch = get_microbatch_fn(batch, loop_cnt)
        # We need to annotate the microbatch sharding as we would do to a batch.
        mbatch = jax.tree_util.tree_map(
            lambda x: with_sharding_constraint(x, PS('dp')),
            mbatch
        )
        (loss, metrics), grad = grad_fn(
            train_state.params,
            mbatch,
            rng_output,
        )
        return loss, grad, metrics

    def per_microbatch_train_step(
        loop_cnt: int,
        state: Tuple[jnp.ndarray,
                     Mapping[str, jnp.ndarray],
                     Optional[flax.core.FrozenDict]]
    ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray],
               Optional[flax.core.FrozenDict]]:
        (loss_accum, grad_accum, metrics_accum) = state
        loss, grad, metrics = calculate_grad(loop_cnt)

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
        return loss_accum, grad_accum, metrics_accum

    # Initialize gradient accumulation loop state.
    loss_accum_init = jnp.zeros((), accum_dtype)
    grad_accum_init = jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape, accum_dtype),
        train_state.params
    )

    _, _, initial_metrics_shape = jax.eval_shape(
        calculate_grad, loop_cnt=0
    )

    metrics_accum_init = {
        k: jnp.zeros((), accum_dtype)
        for k in initial_metrics_shape
    }
    loop_init = (
        loss_accum_init,
        grad_accum_init,
        metrics_accum_init
    )
    loss_accum, grad_accum, metrics_accum = jax.lax.fori_loop(
        0, num_microbatches, per_microbatch_train_step, loop_init
    )

    # normalize loss and gradient by number of tokens
    num_tokens = metrics_accum['n_tokens']
    # only normalize if num_tokens > 0

    def _normalize(loss, grad, num_tokens):
        loss = loss / num_tokens
        grad = jax.tree_map(lambda x: x / num_tokens, grad)
        return loss, grad

    loss_accum, grad_accum = jax.lax.cond(
        num_tokens > 0,
        lambda x: _normalize(*x),
        lambda x: x[:-1],  # noop, return loss, grad
        (loss_accum, grad_accum, num_tokens)
    )

    # Apply the gradients to the model.
    train_state = train_state.apply_gradients(grads=grad_accum)
    metrics = dict(
        loss=loss_accum,
        accuracy=metrics_accum['n_correct'] / metrics_accum['n_tokens'],
        learning_rate=learning_rate_schedule(train_state.step),
        gradient_norm=global_norm(grad_accum),
        param_norm=global_norm(train_state.params),
    )
    return train_state, rng_output, metrics


def eval_step(
    train_state,
    batch,
    rng_output,
    model=None,
):
    loss, metrics = loss_and_metrics(
        train_state.params,
        batch,
        rng_output,
        model=model
    )
    metrics = dict(
        eval_loss=loss,
        eval_accuracy=metrics["n_correct"] / metrics["n_tokens"],
        **{f"eval_{k}": v for k, v in metrics.items()}
    )
    return metrics
