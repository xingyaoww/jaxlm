import math
import jax.numpy as jnp
import numpy as np
import threading
from datasets import Dataset
from queue import Queue
from typing import Iterable


def pad_to_batch_size(
    batch,
    batch_size,
    key_to_pad_val,
    ignore_keys=[]
):
    num_examples = len(batch[list(batch.keys())[0]])
    if num_examples < batch_size:
        pad_len = batch_size - num_examples
        for k, v in batch.items():
            if k == "id":
                assert isinstance(batch[k], list)
                batch[k] = batch[k] + [-1] * pad_len
                continue

            if k in ignore_keys:
                batch[k] = np.array(list(batch[k]) + [None] * pad_len)
                continue

            if not isinstance(v, np.ndarray):
                v = np.array(v)
            batch[k] = np.concatenate(
                [
                    v,
                    np.ones(
                        (pad_len,) + v.shape[1:],
                        dtype=v.dtype,
                    ) * key_to_pad_val[k]
                ],
                axis=0,
            )
    return batch


class DataloaderPrefetchWrapper:
    def __init__(
        self,
        dataloader: Iterable,
        prefetch_size: int = 4,
    ):
        """
        # some testcases
        fake_dl = list(range(6))
        dl_wrapped = DataloaderPrefetchWrapper(iter(fake_dl), prefetch_size=8)

        outputs = []
        for i in dl_wrapped:
            outputs.append(i)
        assert outputs == fake_dl
        """
        self.dataloader: Iterable = dataloader
        self.buffer_size: int = prefetch_size

        self.data_queue: Queue = Queue(maxsize=self.buffer_size)
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_thread_fn,
            daemon=True,
        )
        self.prefetch_thread.start()

    def _prefetch_thread_fn(self):
        while True:
            # load next element (in background)
            try:
                next_element = next(self.dataloader)
            except StopIteration:
                next_element = None
            # it will block if the queue is full
            self.data_queue.put(next_element)

            if next_element is None:
                break

    def __iter__(self):
        return self

    def __next__(self):
        # get next element
        next_element = self.data_queue.get()
        if next_element is None:
            raise StopIteration
        return next_element


def dataloader_impl(
    dataset: Dataset,
    batch_size: int,
    return_idx: bool = False,
    return_jnp_array: bool = False,
):
    """
    Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`, the final batch may be incomplete,
    and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.
    """
    # require shuffle to be done in dataset
    batch_idx = np.arange(len(dataset))

    # dataset should pad the last batch to batch_size
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {
            k: jnp.array(v) if return_jnp_array
            else np.array(v)
            for k, v in batch.items()
        }
        if return_idx:
            yield idx, batch
        else:
            yield batch


def dataloader(*args, **kwargs):
    """
    This is a wrapper around the dataloader_impl that prefetches the next batch.
    """
    return DataloaderPrefetchWrapper(dataloader_impl(*args, **kwargs))
