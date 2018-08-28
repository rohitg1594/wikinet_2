# Various Utility Functions to be used elsewhere
import numpy as np

import string
import re

import sys


RE_WS_PRE_PUCT = re.compile(u'\s+([^a-zA-Z\d])')
RE_WIKI_ENT = re.compile(r'.*wiki\/(.*)')
RE_WS = re.compile('\s+')


def gen_wrapper(gen):
    while True:
        try:
            yield next(gen)
        except StopIteration:
            raise
        except Exception as e:
            print(e)
            pass


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    x_c = x.copy()
    x_c[x_c < 0] = 0
    return x_c


def PCA(x, k=2):
    x_mean = np.mean(x,0)
    x = x - x_mean
    u, s, v = np.linalg.svd(x.T)

    return x @ u[:, :k]


def normal_initialize(dim_0=1000, dim_1=16):
    stdv = 1 / np.sqrt(dim_1)
    gram_embs = np.random.normal(0, scale=stdv, size=(dim_0, dim_1))
    gram_embs[0] = np.zeros(dim_1)

    return gram_embs


def normalize(v):
    if len(v.shape) == 1:
        return v / (np.linalg.norm(v) + 10**-11)
    elif len(v.shape) == 2:
        norm = np.linalg.norm(v, axis=1) + 10**-11
        return v / norm[:, None]
    else:
        print("normalize only accepts arrays of dimensions 1 or 2.")
        sys.exit(1)


def list_line_locations(filename):
    line_offset = []
    offset = 0
    with open(filename, "rb") as f:
        for line in f:
            line_offset.append(offset)
            offset += len(line)
    return line_offset


def reverse_dict(d):

    return {v: k for k, v in d.items()}


def normalise_form(sf):
    sf = sf.lower()
    sf = RE_WS_PRE_PUCT.sub(r'\1', sf)
    sf = RE_WS.sub(' ', sf)
    return sf


def iter_derived_forms(sf):
    yield sf
    yield sf.replace("'s", "")
    yield ''.join(c for c in sf if not c in string.punctuation)

    comma_parts = sf.split(',')[:-1]
    for i in range(len(comma_parts)):
        yield ''.join(comma_parts[:i + 1])
    if comma_parts:
        yield ''.join(comma_parts)

    colon_idx = sf.find(':')
    if colon_idx != -1:
        yield sf[:colon_idx]


def get_normalised_forms(sf):
    return set(normalise_form(f) for f in iter_derived_forms(sf))


def equalize_len(data, max_size):
    d = data.copy()
    l = len(d)
    if l >= max_size:
        return d[:max_size]
    else:
        for _ in range(max_size - l):
            d.append(0)

        return d


def str2bool(v):
    """
    thanks : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    :param v:
    :return:
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False