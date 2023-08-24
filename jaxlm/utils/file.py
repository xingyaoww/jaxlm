# This code is copied and modified from young-geng/mlxu, licensed under MIT License
# Original source: https://github.com/young-geng/mlxu/blob/main/mlxu/utils.py
import logging
import cloudpickle as pickle
import gcsfs


def open_file(path, mode='rb', cache_type='readahead'):
    if path.startswith("gs://"):
        logging.getLogger("fsspec").setLevel(logging.WARNING)
        return gcsfs.GCSFileSystem().open(path, mode, cache_type=cache_type)
    else:
        return open(path, mode)


def save_pickle(obj, path):
    with open_file(path, 'wb') as fout:
        pickle.dump(obj, fout)


def load_pickle(path):
    with open_file(path, 'rb') as fin:
        data = pickle.load(fin)
    return data
