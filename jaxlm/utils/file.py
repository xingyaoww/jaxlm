# This code is copied and modified from young-geng/mlxu, licensed under MIT License
# Original source: https://github.com/young-geng/mlxu/blob/main/mlxu/utils.py
import os
import glob as glob_pkg
import logging
import cloudpickle as pickle
import gcsfs


def open_file(path, mode='rb', cache_type='readahead'):
    path = str(path)
    if path.startswith("gs://"):
        logging.getLogger("fsspec").setLevel(logging.WARNING)
        return gcsfs.GCSFileSystem().open(path, mode, cache_type=cache_type)
    else:
        return open(path, mode)

def glob(path):
    path = str(path)
    if path.startswith("gs://"):
        return gcsfs.GCSFileSystem().glob(path)
    else:
        return glob_pkg.glob(path)

def exists(path):
    path = str(path)
    if path.startswith("gs://"):
        return gcsfs.GCSFileSystem().exists(path)
    else:
        return os.path.exists(path)

def mkdir(path):
    path = str(path)
    if path.startswith("gs://"):
        return gcsfs.GCSFileSystem().mkdir(path)
    else:
        return os.mkdir(path)

def save_pickle(obj, path):
    with open_file(path, 'wb') as fout:
        pickle.dump(obj, fout)


def load_pickle(path):
    with open_file(path, 'rb') as fin:
        data = pickle.load(fin)
    return data
