#!/bin/bash

# Files
VOCAB_FILE='gs://xingyao6/llama-code/data/model/llama-ckpt/tokenizer.model'
INIT_CKPT='params::/root/llama-code/data/model/easylm/llama-2-13b-chat.easylm_stream.ckpt'
OUTPUT_DIR='gs://xingyao6/llama-code/data/model/easylm/llama-code-pretrain'

# Config
BATCH_SIZE_PER_DEVICE=4
N_MICROBATCHES=4
SAVE_FREQ=100
N_STEPS=1000

python3 jaxlm/examples/run_clm_llama.py \
    --seed=42 \
    --mesh_dim='1,8,-1' \
    --dataset_name='wikitext' \
    --dataset_config_name='wikitext-2-raw-v1' \
    --text_column_name='text' \
    --block_size=4096 \
    --load_llama_config='13b' \
    --vocab_file=$VOCAB_FILE \
    --load_checkpoint=$INIT_CKPT \
    --batch_size_per_device=$BATCH_SIZE_PER_DEVICE \
    --num_microbatches=$N_MICROBATCHES \
    --block_size=4096 \
    --total_steps=$N_STEPS \
    --log_freq=5 \
    --save_freq=$SAVE_FREQ
