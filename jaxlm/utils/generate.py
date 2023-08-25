# This code is modified from young-geng/EasyLM, licensed under Apache License 2.0
# Original source: https://github.com/young-geng/EasyLM/blob/main/EasyLM/models/llama/llama_serve.py

from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
import optax
from transformers import GenerationConfig, FlaxLogitsProcessorList

from jaxlm.utils.jax import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules, tree_apply,
    set_random_seed, get_float_dtype_by_name, make_shard_and_gather_fns,
    with_sharding_constraint, FlaxTemperatureLogitsWarper
)

def forward_loglikelihood(params, rng, batch):
    batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
    rng_generator = JaxRNG(rng)
    input_tokens = batch['input_tokens']
    output_tokens = batch['output_tokens']
    input_mask = batch['input_mask']
    output_mask = batch['output_mask']

    logits = hf_model.module.apply(
        params, input_tokens, attention_mask=input_mask,
        deterministic=True, rngs=rng_generator(llama_config.rng_keys()),
    ).logits
    # if llama_config.n_real_tokens is not None:
    #   logits = logits.at[:, :, llama_config.n_real_tokens:].set(-1e8)
    loglikelihood = -optax.softmax_cross_entropy_with_integer_labels(
        logits, output_tokens
    )
    loglikelihood = jnp.sum(loglikelihood * output_mask, axis=-1)
    match_count = jnp.sum(
        (jnp.argmax(logits, axis=-1) == output_tokens) * output_mask,
        axis=-1
    )
    total = jnp.sum(output_mask, axis=-1)
    is_greedy = match_count == total
    return loglikelihood, is_greedy, rng_generator()


def forward_generate(
    params,
    rng,
    batch,
    temperature,
    seq_length,
    hf_model,
    tokenizer,
    top_k=1,
    top_p=1.0,
):
    batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
    rng_generator = JaxRNG(rng)
    input_length = batch['input_tokens'].shape[1]

    if temperature == 0:
        logits_processor = None
        do_sample = False
    else:
        logits_processor = FlaxLogitsProcessorList(
            [FlaxTemperatureLogitsWarper(temperature)]
        )
        do_sample = True

    output = hf_model.generate(
        batch['input_tokens'],
        attention_mask=batch['attention_mask'],
        params=params['params'],
        prng_key=rng_generator(),
        logits_processor=logits_processor,
        generation_config=GenerationConfig(
            max_new_tokens=seq_length - input_length,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
            num_beams=1,
            top_k=top_k,
            top_p=top_p,
        )
    ).sequences[:, batch['input_tokens'].shape[1]:]
    return output, rng_generator()
