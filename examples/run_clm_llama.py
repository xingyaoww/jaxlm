import jax.numpy as jnp
import jax
import flax
from jax.experimental.compilation_cache import compilation_cache as cc
from typing import Mapping, Tuple, Optional
from flax.training.train_state import TrainState
from jax.sharding import PartitionSpec as PS
from jax.experimental.pjit import pjit

import pprint
import re
import logging
import argparse
import subprocess
from itertools import chain
from functools import partial
from tqdm import tqdm, trange
from datasets import load_dataset, load_from_disk

from jaxlm.models.llama.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLMModule
)
from jaxlm.utils.training import (
    train_step_microbatched, eval_step,
)
from jaxlm.jaxlm.optimizers import AdamWOptimizerFactory
from jaxlm.utils.checkpoint import StreamingCheckpointer
import jaxlm.utils.file as fileutils
from jaxlm.utils.data import (
    pad_to_batch_size,
    dataloader
)
from jaxlm.utils.jax import (
    JaxRNG,
    set_random_seed,
    next_global_rng,
    init_rng,
    match_partition_rules,
    get_float_dtype_by_name,
    average_metrics,
    make_shard_and_gather_fns,
)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger("jaxlm")


cc.initialize_cache("/tmp/jax_cache")


def main(argv):
    if FLAGS.initialize_jax_distributed:
        jax.distributed.initialize()

    set_random_seed(FLAGS.seed)

    # ==== Load information required to recover model & data from checkpoint (if needed) ====
    # Decide whether to load the latest checkpoint and/or dataset state
    def get_latest_checkpoint(output_dir):
        # Detect Checkpoint
        path_prefix = f"{output_dir}/streaming_train_state"
        output = subprocess.check_output(
            f"gcloud storage ls {path_prefix}*".split()
        ).decode('utf-8')
        checkpoints = re.findall(r'streaming_train_state_(\d+)', output)
        if len(checkpoints) == 0:
            return None, None
        checkpoints = sorted([int(x) for x in checkpoints])
        print(f"Found {len(checkpoints)} existing checkpoints: {checkpoints}")
        latest_checkpoint_step = max(checkpoints)
        checkpoint_path = f"{path_prefix}_{latest_checkpoint_step}"

        # Detect Dataset State
        dataset_path_pkl = f"{output_dir}/dataset_{latest_checkpoint_step}.pkl"
        output = subprocess.check_output(
            f"gcloud storage ls {dataset_path_pkl}".split()
        ).decode('utf-8')
        dataset_path = '' if len(output) == 0 else dataset_path_pkl

        return checkpoint_path, dataset_path

    latest_checkpoint, latest_dataset_state_path = get_latest_checkpoint()
    latest_checkpoint = "trainstate::" + latest_checkpoint
    if latest_checkpoint is not None:
        FLAGS.load_checkpoint = latest_checkpoint
        print(
            f"Latest checkpoint found: {latest_checkpoint}; It will overwrite the input FLAGS.load_checkpoint argument.")

    if latest_dataset_state_path is not None and latest_dataset_state_path != '':
        FLAGS.load_dataset_state = latest_dataset_state_path
        print(
            f"Latest dataset state found: {latest_dataset_state_path}; It will overwrite the input FLAGS.load_dataset_state argument.")

    # ==== Load Model and Tokenizer ====
    tokenizer = LLaMAConfig.get_tokenizer(FLAGS.vocab_file)
    if FLAGS.load_llama_config != '':
        llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
    llama_config.update(dict(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ))
    if llama_config.vocab_size < tokenizer.vocab_size:
        llama_config.update(dict(vocab_size=tokenizer.vocab_size))
    rng_keys = llama_config.rng_keys()
    model = FlaxLLaMAForCausalLMModule(
        llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )

    # ==== Load Dataset ====
    KEY_TO_PAD_ID = {
        "input_ids": tokenizer.pad_token_id,
        "attention_mask": 0,
        "labels": -100,
        "loss_weights": 0,
    }
    # Load dataset from huggingface datasets
    if not FLAGS.load_from_disk:  # Load from huggingface datasets and do preprocessing
        raw_datasets = load_dataset(
            FLAGS.dataset_name, FLAGS.dataset_config_name)
        if FLAGS.do_validation and "validation" not in raw_datasets.keys():
            splited_dataset = raw_datasets["train"].train_test_split(
                test_size=FLAGS.validation_split_percentage / 100.0
            )
            logger.info(
                f"Splitting {FLAGS.validation_split_percentage}% of training data as validation set: val size {splited_dataset['test'].shape}, train size {splited_dataset['train'].shape}"
            )
            raw_datasets["validation"] = splited_dataset["test"]
            raw_datasets["train"] = splited_dataset["train"]

        # Tokenize dataset
        column_names = raw_datasets["train"].column_names

        def tokenize_function(examples):
            return tokenizer(examples[FLAGS.text_column_name])

        if jax.process_index() == 0:  # Only do this on the first process
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=FLAGS.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not FLAGS.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        if FLAGS.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 1024:
                logger.warning(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default with `--block_size xxx`."
                )
            block_size = 1024
        else:
            if FLAGS.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({FLAGS.block_size}) is larger than the maximum length for the model"
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(FLAGS.block_size, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[FLAGS.text_column_name])

            def _pad_to_block_size(l: list, pad_with):
                assert len(l) <= block_size
                if len(l) < block_size:
                    l += [pad_with] * (block_size - len(l))
                return l

            result = {
                k: [
                    _pad_to_block_size(
                        t[i:i + block_size],
                        KEY_TO_PAD_ID[k]
                    )
                    for i in range(0, total_length, block_size)
                ]
                for k, t in concatenated_examples.items()
            }

            result["labels"] = result["input_ids"].copy()
            # We only calculate loss on non-padded tokens.
            result["loss_weights"] = result["labels"] != tokenizer.pad_token_id
            return result

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=FLAGS.preprocessing_num_workers,
            load_from_cache_file=not FLAGS.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    else:
        lm_datasets = load_from_disk(FLAGS.load_from_disk)

    # Pad to batch size
    batch_size = FLAGS.batch_size_per_device * jax.device_count()
    lm_datasets = lm_datasets.map(
        partial(pad_to_batch_size, batch_size=batch_size,
                key_to_pad_val=KEY_TO_PAD_ID),
        batched=True,
        batch_size=batch_size,
        num_proc=FLAGS.preprocessing_num_workers,
        load_from_cache_file=not FLAGS.overwrite_cache,
        desc=f"Padding to batch size {batch_size}",
    )

    train_dataset = lm_datasets["train"]
    if FLAGS.do_validation:
        val_dataset = lm_datasets["validation"]

    # ==== Load Optimizer ====
    optimizer, optimizer_info = AdamWOptimizerFactory.get_optimizer(
        init_lr=0.0,
        lr=0.01,
        end_lr=0.001,
        lr_warmup_steps=2000,
        lr_decay_steps=500000,
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, FLAGS.block_size), dtype=jnp.int32),
            position_ids=jnp.zeros((4, FLAGS.block_size), dtype=jnp.int32),
            attention_mask=jnp.ones((4, FLAGS.block_size), dtype=jnp.int32),
            rngs=rng_generator(rng_keys),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    # ==== Load Checkpointer ====
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )

    train_state_shapes = jax.eval_shape(init_fn, next_global_rng())
    train_state_partition = match_partition_rules(
        LLaMAConfig.get_partition_rules(), train_state_shapes
    )

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    print(f"Number of microbatches: {FLAGS.num_microbatches}")
    sharded_train_step = pjit(
        partial(train_step_microbatched,
                model=model,
                num_microbatches=FLAGS.num_microbatches,
                learning_rate_schedule=optimizer_info.get(
                    "learning_rate_schedule", None)
                ),
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        partial(eval_step, model=model),
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            flags=vars(FLAGS),
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            milestone=milestone,
        )

    mesh = LLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        train_state, restored_params = None, None

        if FLAGS.load_checkpoint != '':
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_global_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(
                restored_params)
            del restored_params

        start_step = int(jax.device_get(train_state.step))

        if FLAGS.save_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_global_rng()
        step_counter = trange(0, FLAGS.total_steps, ncols=0)

        for step, batch in zip(step_counter, train_dataset):
            # Skip steps until start_step
            if step < start_step:
                continue

            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, batch, rng_output=sharded_rng(rng_keys),
            )

            if FLAGS.do_validation and step % FLAGS.eval_freq == 0:
                eval_metric_list = []
                eval_rng = init_rng(FLAGS.seed)
                for eval_batch in val_dataset:
                    eval_metrics = sharded_eval_step(
                        train_state, eval_batch,
                        rng_output=eval_rng(rng_keys),

                    )
                    eval_metric_list.append(eval_metrics)
                metrics.update(average_metrics(eval_metric_list))

            if step % FLAGS.log_freq == 0:
                log_metrics = {"step": step}
                log_metrics.update(metrics)
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            elif FLAGS.save_freq > 0 and (step + 1) % FLAGS.save_freq == 0:
                save_checkpoint(train_state, milestone=True)

        save_checkpoint(train_state, milestone=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    # === Distributed Training ===
    parser.add_argument('--initialize_jax_distributed', action='store_true')
    # https://github.com/young-geng/EasyLM/blob/main/docs/parallelism.md
    parser.add_argument('--mesh_dim', type=str, default='1,-1,1',
                        help="(data, FSDP, model) parallelism dimensions")

    # === Dataset ===
    parser.add_argument('--dataset_name', type=str, default='wikitext')
    parser.add_argument('--dataset_config_name', type=str,
                        default='wikitext-2-raw-v1')
    parser.add_argument('--text_column_name', type=str, default='text')
    parser.add_argument('--validation_split_percentage', type=int, default=5)
    parser.add_argument('--load_from_disk', action='store_true')
    parser.add_argument('--do_validation', action='store_true')
    parser.add_argument('--preprocessing_num_workers', type=int, default=8)
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument('--block_size', type=int, default=4096,
                        help="Max sequence length for packing.")

    # === Model ===
    parser.add_argument('--load_llama_config', type=str,
                        default='7b', help="available: 7b, 13b")
    parser.add_argument('--vocab_file', type=str, default='')
    parser.add_argument('--load_checkpoint', type=str, default='')

    # === Training ===
    parser.add_argument('--batch_size_per_device', type=int, default=1)
    parser.add_argument('--num_microbatches', type=int, default=1)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--total_steps', type=int, default=10000)
    parser.add_argument('--dtype', type=str, default='bf16')
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--save_freq', type=int, default=0)
    parser.add_argument('--eval_freq', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default=None)
    FLAGS = parser.parse_args()
    main()
