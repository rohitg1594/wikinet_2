# Various Utility Functions to be used elsewhere
import numpy as np

import string
import re
import sys
from os.path import join
import logging
import pickle
import random
from collections import defaultdict

import torch
from torch.nn import DataParallel
import torch.nn as nn

from src.models.models import Models


use_cuda = torch.cuda.is_available()
RE_WS_PRE_PUCT = re.compile(u'\s+([^a-zA-Z\d])')
RE_WIKI_ENT = re.compile(r'.*wiki\/(.*)')
RE_WS = re.compile('\s+')

logger = logging.getLogger(__name__)


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
    """Initialize with normal distribution of std = 1 / sqrt(dim_1). Set O index to all zeros."""
    stdv = 1 / np.sqrt(dim_1)
    embs = np.random.normal(0, scale=stdv, size=(dim_0, dim_1))
    embs[0] = np.zeros(dim_1)

    return embs


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

    if sf.startswith('The') or sf.startswith('the'):
        yield sf[4:]

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
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def get_model(args, yamada_model=None, gram_embs=None, ent_embs=None, word_embs=None, init=None):
    """Based on parameters in args, initialize and return appropriate model."""

    kwargs = {'args': args,
              'gram_embs': gram_embs,
              'ent_embs': ent_embs,
              'word_embs': word_embs,
              'W': yamada_model['W'],
              'b': yamada_model['b']}

    weighing_linear = torch.Tensor(ent_embs.shape[1] + gram_embs.shape[1], 1)
    torch.nn.init.eye(weighing_linear)
    kwargs['weighing_linear'] = weighing_linear

    if init == 'pre_trained':
        logger.info("Loading mention and ent mention embs from {}".format(args.init_mention_model))
        with open(args.init_mention_model, 'rb') as f:
            ckpt = torch.load(f, map_location='cpu')
        mention_embs = ckpt['state_dict']['mention_embs.weight']
        ent_mention_embs = ckpt['state_dict']['ent_mention_embs.weight']

    elif init == 'pca':
        logger.info("Loading mention and ent mention embs from yamada pca at {}.".format(args.init_mention_model))
        with open(join(args.data_path, 'yamada', args.init_mention_model), 'rb') as f:
            d = pickle.load(f)
        mention_embs = torch.from_numpy(d['word'])
        ent_mention_embs = torch.from_numpy(d['ent'])

    else:
        mention_embs = torch.from_numpy(np.random.normal(loc=0, scale=args.init_stdv,
                                                         size=(word_embs.shape[0], args.mention_word_dim)))
        ent_mention_embs = torch.from_numpy(np.random.normal(loc=0, scale=args.init_stdv,
                                                             size=(ent_embs.shape[0], args.ent_mention_dim)))
    mention_embs[0] = 0
    ent_mention_embs[0] = 0

    if args.gram_type == 'bigram':
        kernel = 2
    else:
        kernel = 3
    conv_weights = torch.Tensor(mention_embs.shape[1], mention_embs.shape[1], kernel)
    if init == 'xavier_uniform':
        nn.init.xavier_uniform(conv_weights)
    elif init == 'xavier_normal':
        nn.init.xavier_normal(conv_weights)
    kwargs['conv_weights'] = conv_weights
    kwargs['mention_embs'] = mention_embs
    kwargs['ent_mention_embs'] = ent_mention_embs

    model_type = getattr(Models, args.model_name)
    model = model_type(**kwargs)

    if args.use_cuda:
        model = send_to_cuda(args.device, model)
    logger.info('{} Model created.'.format(model_type.__name__))

    return model


def send_to_cuda(device, model):
    if isinstance(device, tuple):
        model = DataParallel(model, device)
    else:
        model.cuda(device)

    return model


def get_absolute_pos(word_sequences):
    batch = np.zeros_like(word_sequences, dtype=np.int64)
    for i, word_seq in enumerate(word_sequences):
        start_idx = 1
        for j, pos in enumerate(word_seq):
            if int(pos) == 0:
                batch[i, j] = 0
            else:
                batch[i, j] = start_idx
                start_idx += 1
    return torch.from_numpy(batch)


def probe(d, n=10):

    for i, (k, v) in enumerate(d.items()):
        if i == n:
            break
        print(k, v)


def check_errors(I, gold, gram_indices, rev_ent_dict, rev_gram_dict, ks):
    errors = defaultdict(list)

    for j, k in enumerate(ks):
        for i in range(I.shape[0]):
            if gold[i] not in I[i, :k]:
                errors[k].append((i, gold[i], I[i]))

    for k, errors in errors.items():
        print("Top {} errors:".format(k))
        mask = random.sample(range(len(errors)), 10)
        for i in mask:
            mention_idx, gold_id, predictions_id = errors[i]
            mention_tokens = gram_indices[mention_idx]
            predictions = ','.join([rev_ent_dict.get(ent_id, '') for ent_id in predictions_id][:10])

            mention_grams = []
            for token_idx, token in enumerate(mention_tokens):
                if token == 0:
                    break
                elif token in rev_gram_dict:
                    mention_grams.append(rev_gram_dict[token][0])
            mention = ''.join(mention_grams)
            if token_idx > 0:
                last_gram = rev_gram_dict.get(mention_tokens[token_idx-1], '')
                if len(last_gram) > 1:
                    mention += last_gram[1:]

            print('{}|{}|{}'.format(mention, rev_ent_dict.get(gold_id, ''), predictions))
        print()


def eval_ranking(I, gold, ks):
    topks = np.zeros(len(ks))

    for j, k in enumerate(ks):
        for i in range(I.shape[0]):
            if gold[i] in I[i, :k]:
                topks[j] += 1

    topks /= I.shape[0]

    # Mean Reciprocal Rank
    ranks = []
    for i in range(I.shape[0]):
        index = np.where(gold[i] == I[i])[0] + 1
        if not index:
            ranks.append(1 / I.shape[1])
        else:
            ranks.append(1 / index)
    mrr = np.mean(np.array(ranks))

    if not isinstance(mrr, float):
        mrr = mrr[0]

    return topks[0], topks[1], topks[2], mrr

