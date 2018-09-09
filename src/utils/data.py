# Utility functions to load and save data
import torch
import numpy as np

import os
from os.path import join
import sys
import logging

import pickle

from src.utils.utils import normalize, reverse_dict
from src.conll.pershina import PershinaExamples

logger = logging.getLogger(__name__)


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


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def conll_to_wiki(data, rev_word_dict):
    """Converts conll data from pershina examples to wikipedia training files to work in combined dataloader."""
    res = []
    for words, conll_examples in data:
        word_ids = [rev_word_dict.get(word) for word in words if word in rev_word_dict]
        wiki_examples = [(mention, cands[0]) for mention, cands in conll_examples]
        res.append((word_ids, wiki_examples))
    return res


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


def load_stats(args, yamada_model):
    priors = pickle_load(join(args.data_path, "yamada", "ent_priors.pickle"))
    conditionals = pickle_load(join(args.data_path, "yamada", "ent_conditionals.pickle"))
    ent2index = pickle_load(join(args.data_path, "yamada", "yamada_ent2index.pickle"))
    index2ent = reverse_dict(ent2index)
    ent_dict = yamada_model['ent_dict']
    ent_rev = reverse_dict(ent_dict)

    ent_priors = {}
    for ent_index, p in priors.items():
        ent_str = index2ent[ent_index]
        if ent_str in ent_dict:
            ent_id = ent_dict[ent_str]
            ent_priors[ent_id] = p

    ent_conditionals = {}
    for mention, cond_dict in conditionals.items():
        orig_cond_dict = {}
        for ent_id, p_m in cond_dict.items():
            if ent_id in ent_rev:
                ent_str = ent_rev[ent_id]
                if ent_str in ent_dict:
                    orig_ent_id = ent_dict[ent_str]
                    orig_cond_dict[orig_ent_id] = p_m
        if len(orig_cond_dict) > 0:
            ent_conditionals[mention] = orig_cond_dict

    return priors, conditionals


def load_data(args, yamada_model):
    if args.proto_data:
        train_data, dev_data = pickle_load(join(args.data_path, 'training_files', 'proto.pickle'))
        test_data = []
    else:
        if args.data_type == 'wiki':
            logger.info("Loading Wikipedia Training data.")
            data = []
            for i in range(args.num_shards):
                data.extend(pickle_load(join(args.data_path, 'training_files', 'data_{}.pickle'.format(i))))

            train_data = []
            dev_data = []
            test_data = []
            for d in data:
                if len(train_data) == args.train_size:
                    break
                r = np.random.random()
                if r < 0.90:
                    train_data.append(d)

                elif 0.9 < r < 0.95:
                    dev_data.append(d)
                else:
                    test_data.append(d)

        elif args.data_type == 'conll':
            logger.info("Loading Pershina Training data.")
            pershina = PershinaExamples(args, yamada_model)
            train_data, dev_data, test_data = pershina.get_training_examples()
            rev_word_dict = reverse_dict(yamada_model['word_dict'])
            train_data, dev_data, test_data = conll_to_wiki(train_data, rev_word_dict), \
                                              conll_to_wiki(dev_data, rev_word_dict), conll_to_wiki(test_data,
                                                                                                    rev_word_dict)
        else:
            logger.error("Data type {} not recognized, choose between [wiki, conll]".format(args.data_type))
            sys.exit(1)

    return train_data, dev_data, test_data
