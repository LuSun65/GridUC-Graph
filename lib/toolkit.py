import os
import pickle
import logging
import time
import sys
import random
import numpy as np


def tic():
    return time.time()


def toc(start_time):
    return time.time() - start_time


class _Std2Log:
    def __init__(self, fn):
        self.fn = fn
        self.buf = ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            line = line.rstrip()
            if line:
                self.fn("%s", line)

    def flush(self):
        if self.buf.strip():
            self.fn("%s", self.buf.strip())
        self.buf = ""


def setlog(log_file: str, overwrite: bool = False):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(message)s')

    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = logging.FileHandler(log_file, mode='w' if overwrite else 'a', encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    sys.stdout = _Std2Log(logger.info)
    sys.stderr = _Std2Log(logger.error)


def set_rnd_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def save_pkl(obj, path: str):
    parent_dir = os.path.dirname(path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)
