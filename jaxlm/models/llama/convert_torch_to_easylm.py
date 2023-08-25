# This script converts the standrd LLaMA PyTorch checkpoint released by Meta
# to the EasyLM checkpoint format. The converted checkpoint can then be loaded
# by EasyLM for fine-tuning or inference.

# This script is largely borrow from https://github.com/Sea-Snell/JAX_llama

# This code is copied from young-geng/EasyLM, licensed under Apache License 2.0
# Original source: https://github.com/young-geng/EasyLM/tree/main/EasyLM

from pathlib import Path
import os
import json
import numpy as np
import torch
import flax
import argparse

import jaxlm.utils.file as fileutils
from jaxlm.utils.checkpoint import StreamingCheckpointer


def main(argv):
    ckpt_paths = sorted(fileutils.glob(
        os.path.join(FLAGS.checkpoint_dir, "*.pth")))
    ckpt_paths = [Path(ckpt_path) for ckpt_path in ckpt_paths]
    assert len(ckpt_paths) > 0, "No checkpoint found in %s" % (
        FLAGS.checkpoint_dir)

    ckpts = {}
    for i, ckpt_path in enumerate(ckpt_paths):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        ckpts[int(ckpt_path.name.split('.', maxsplit=2)[1])] = checkpoint
    ckpts = [ckpts[i] for i in sorted(list(ckpts.keys()))]

    with fileutils.open_file(os.path.join(FLAGS.checkpoint_dir, "params.json"), "r") as f:
        params = json.loads(f.read())


    if ckpts[0]['tok_embeddings.weight'].dtype == torch.bfloat16:
        print("Warning: the model is trained with bfloat16, which will be converted to float32.")

    def _torch_to_numpy(t):
        # cast to float32 to avoid "Got unsupported ScalarType BFloat16" issue
        return t.to(torch.float32).numpy()

    jax_weights = {
        'transformer': {
            'wte': {'embedding': np.concatenate([
                _torch_to_numpy(ckpt['tok_embeddings.weight']) for ckpt in ckpts
            ], axis=1)},
            'ln_f': {
                'kernel': _torch_to_numpy(ckpts[0]['norm.weight'])
            },
            'h': {
                '%d' % (layer): {
                    'attention': {
                        'wq': {'kernel': np.concatenate([
                            _torch_to_numpy(
                                ckpt['layers.%d.attention.wq.weight' % (layer)])
                            for ckpt in ckpts
                        ], axis=0).transpose()},
                        'wk': {'kernel': np.concatenate([
                            _torch_to_numpy(
                                ckpt['layers.%d.attention.wk.weight' % (layer)])
                            for ckpt in ckpts], axis=0).transpose()
                        },
                        'wv': {'kernel': np.concatenate([
                            _torch_to_numpy(
                                ckpt['layers.%d.attention.wv.weight' % (layer)])
                            for ckpt in ckpts
                        ], axis=0).transpose()},
                        'wo': {'kernel': np.concatenate([
                            _torch_to_numpy(
                                ckpt['layers.%d.attention.wo.weight' % (layer)])
                            for ckpt in ckpts
                        ], axis=1).transpose()},
                    },
                    'feed_forward': {
                        'w1': {'kernel': np.concatenate([
                            _torch_to_numpy(
                                ckpt['layers.%d.feed_forward.w1.weight' % (layer)])
                            for ckpt in ckpts
                        ], axis=0).transpose()},
                        'w2': {'kernel': np.concatenate([
                            _torch_to_numpy(
                                ckpt['layers.%d.feed_forward.w2.weight' % (layer)])
                            for ckpt in ckpts
                        ], axis=1).transpose()},
                        'w3': {'kernel': np.concatenate([
                            _torch_to_numpy(
                                ckpt['layers.%d.feed_forward.w3.weight' % (layer)])
                            for ckpt in ckpts
                        ], axis=0).transpose()},
                    },
                    'attention_norm': {
                        'kernel':
                        _torch_to_numpy(
                            ckpts[0]['layers.%d.attention_norm.weight' % (layer)])
                    },
                    'ffn_norm': {
                        'kernel':
                        _torch_to_numpy(
                            ckpts[0]['layers.%d.ffn_norm.weight' % (layer)])
                    },
                }
                for layer in range(params['n_layers'])},
        },
        'lm_head': {'kernel': np.concatenate([
            _torch_to_numpy(ckpt['output.weight'])
            for ckpt in ckpts
        ], axis=0).transpose()},
    }

    if not fileutils.exists(os.path.dirname(FLAGS.output_file)):
        fileutils.mkdir(os.path.dirname(FLAGS.output_file))
    if FLAGS.streaming:
        StreamingCheckpointer.save_train_state_to_file(
            jax_weights, FLAGS.output_file + ".easylm_stream.ckpt"
        )
    else:
        with fileutils.open_file(FLAGS.output_file + ".flax_msgpack", 'wb') as fout:
            fout.write(flax.serialization.msgpack_serialize(
                jax_weights, in_place=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='')
    parser.add_argument('--output_file', type=str, default='')
    parser.add_argument('--streaming', type=bool, default=True)
    FLAGS = parser.parse_args()
    main(FLAGS)
