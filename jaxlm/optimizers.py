# This code is copied from young-geng/EasyLM, licensed under Apache License 2.0
# Original source: https://github.com/young-geng/EasyLM/tree/main/EasyLM
from typing import NamedTuple, Callable
import jax
import jax.numpy as jnp
import optax


def create_learning_rate_fn(
    init_lr: float,
    lr: float,
    end_lr: float,
    lr_warmup_steps: int,
    lr_decay_steps: int,
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    warmup_fn = optax.linear_schedule(
        init_value=init_lr, end_value=lr, transition_steps=lr_warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=lr, end_value=end_lr, transition_steps=lr_decay_steps
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[lr_warmup_steps]
    )
    return schedule_fn


class AdamWOptimizerFactory(object):
    """AdamW optimizer with linear schedule."""

    @classmethod
    def get_optimizer(
        cls,
        init_lr: float = 0.0,
        lr: float = 0.01,
        end_lr: float = 0.001,
        lr_warmup_steps: int = 2000,
        lr_decay_steps: int = 500000,
        b1: float = 0.9,
        b2: float = 0.95,
        clip_gradient: float = 1.0,
        weight_decay: float = 1e-4,
        bf16_momentum: bool = False,
        multiply_by_parameter_scale: bool = False,
        weight_decay_mask=None,
    ):
        learning_rate_schedule = create_learning_rate_fn(
            init_lr=init_lr,
            lr=lr,
            end_lr=end_lr,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
        )

        optimizer_info = dict(
            learning_rate_schedule=learning_rate_schedule,
        )

        if multiply_by_parameter_scale:
            optimizer = optax.chain(
                optax.clip_by_global_norm(clip_gradient),
                optax.adafactor(
                    learning_rate=learning_rate_schedule,
                    multiply_by_parameter_scale=True,
                    momentum=b1,
                    decay_rate=b2,
                    factored=False,
                    clipping_threshold=None,
                    dtype_momentum=jnp.bfloat16 if bf16_momentum else jnp.float32,
                ),
                optax_add_scheduled_weight_decay(
                    lambda step: -learning_rate_schedule(step) * weight_decay,
                    weight_decay_mask,
                ),
            )
        else:
            optimizer = optax.chain(
                optax.clip_by_global_norm(clip_gradient),
                optax.adamw(
                    learning_rate=learning_rate_schedule,
                    weight_decay=weight_decay,
                    b1=b1,
                    b2=b2,
                    mask=weight_decay_mask,
                    mu_dtype=jnp.bfloat16 if bf16_momentum else jnp.float32,
                ),
            )

        return optimizer, optimizer_info


class OptaxScheduledWeightDecayState(NamedTuple):
    count: jnp.DeviceArray


def optax_add_scheduled_weight_decay(schedule_fn, mask=None):
    """Apply weight decay with schedule."""

    def init_fn(params):
        del params
        return OptaxScheduledWeightDecayState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError("Params cannot be None for weight decay!")

        weight_decay = schedule_fn(state.count)
        updates = jax.tree_util.tree_map(
            lambda g, p: g + weight_decay * p, updates, params
        )
        return updates, OptaxScheduledWeightDecayState(
            count=optax.safe_int32_increment(state.count)
        )

    if mask is not None:
        return optax.masked(optax.GradientTransformation(init_fn, update_fn), mask)
    return optax.GradientTransformation(init_fn, update_fn)
