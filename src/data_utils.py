# Utility functions to load and save data
import torch
import numpy as np

import os
import sys
import shutil

import pickle

from src.utils import normalize


def load_vocab(vocab_path, max_vocab=-1, plus_one=False):
    d = {}
    with open(vocab_path, 'r') as f:
        for i, line in enumerate(f):
            if i == max_vocab:
                break
            key, id = line.rstrip().split('\t')
            value = int(id) if int(id) > 0 else -int(id)
            if plus_one:
                value += 1
            d[key] = value
    return d


def pickle_load(path):
    assert os.path.exists(path)

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_yamada(path):
    """Loads yamada model from ntee and returns dictionary of W, word_embs, ent_embs,
    word_dict, end_dict, W and b"""

    with open(path, 'rb') as f:
        model_dict = pickle.load(f, encoding='bytes')

    orig_word_emb = model_dict[b'word_emb']
    num_words, word_dim = orig_word_emb.shape
    orig_word_emb = np.concatenate((np.zeros((1, word_dim)), orig_word_emb), axis=0).astype(np.float32)

    orig_ent_emb = model_dict[b'ent_emb']
    orig_ent_emb = np.concatenate((np.zeros((1, word_dim)), orig_ent_emb), axis=0).astype(np.float32)
    orig_ent_emb = normalize(orig_ent_emb)

    orig_word_trie = model_dict[b'word_dict']
    orig_ent_trie = model_dict[b'ent_dict']
    orig_b = model_dict[b'b']
    orig_W = model_dict[b'W']

    orig_ent_dict = {}
    orig_ent_keys = orig_ent_trie.keys()

    for i, k in enumerate(orig_ent_keys):
        k_id = orig_ent_trie.key_id(k)
        k_id += 1
        k = k.replace(' ', '_')
        orig_ent_dict[k] = k_id

    orig_word_dict = {}
    orig_word_keys = orig_word_trie.keys()

    for i, k in enumerate(orig_word_keys):
        k_id = orig_word_trie.key_id(k)
        k_id += 1
        orig_word_dict[k] = k_id

    return {'word_dict': orig_word_dict,
            'ent_dict': orig_ent_dict,
            'word_emb': orig_word_emb,
            'ent_emb': orig_ent_emb,
            'W': orig_W,
            'b': orig_b}
