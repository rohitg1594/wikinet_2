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
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn

from src.models.models import Models
from src.utils.data import load_gensim


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


def equalize_len_w_eot(data, max_size, eot=None):
    l = len(data)
    arr = np.zeros(max_size, dtype=np.int64)

    if l >= max_size:
        arr[:max_size] = data[:max_size]
        arr[max_size - 1] = eot
    else:
        arr[:l] = data
        arr[l] = eot

    return arr


def str2bool(v):
    """
    thanks : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def get_model(**kwargs):
    """Based on parameters in args, initialize and return appropriate model."""

    args = kwargs['args']

    model_type = getattr(Models, args.model_name)
    model = model_type(**kwargs)

    if args.use_cuda:
        model = send_to_cuda(args.device, model)
    logger.info('{} Model created.'.format(model_type.__name__))

    return model


def send_to_cuda(device, model):
    if isinstance(device, tuple):
        model = DataParallel(model, device)
        model = model.to(device[0])
    else:
        model = model.cuda(device)

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


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def create_arr(strs, max_char, char_dict, ent2id=None):
    dim_0 = len(strs) + 1 if ent2id else len(strs)
    arr = np.zeros((dim_0, max_char), dtype=np.int64)
    repl = ' ' if ent2id else ' '

    for i, s in enumerate(strs):
        char_ids = [char_dict[char] for char in list(s.replace('_', repl))]
        index = ent2id.get(s, 0) if ent2id else i
        arr[index] = equalize_len_w_eot(char_ids, max_char, char_dict['EOT'])

    return arr


def decode(arr, char_dict):
    s = ''
    for c_id in arr:
        if c_id:
            s += char_dict[int(c_id)]
    return s


def mse(input, target):
    b = input.shape[0]

    return ((input - target) * (input - target)).sum() / b


def get_context_embs(data_path=None, emb_option=None, yamada_model=None):

    num_word, word_dim = yamada_model['word_dim'].shape
    num_ent, ent_dim = yamada_model['ent_dim'].shape

    if emb_option == 'random':
        logger.info(f"Initializing context embs randomly.....")
        word_embs = normal_initialize(num_word, word_dim)
        ent_embs = normal_initialize(num_ent, ent_dim)
        W = normal_initialize(word_dim, ent_dim)
        b = np.random.randn(ent_dim)
    elif 'w2v' in emb_option:
        logger.info(f"Loading context embs from {args.init_context_emb}.....")
        ent_embs, word_embs = load_gensim(data_path, model_dir=emb_option, yamada_model=yamada_model)
        W = normal_initialize(word_dim, ent_dim)
        b = np.random.randn(ent_dim)
    elif emb_option == 'yamada':
        logger.info("Loading context embs from yamada model.....")
        ent_embs = yamada_model['ent_emb']
        word_embs = yamada_model['word_emb']
        W = yamada_model['W']
        b = yamada_model['b']
    elif emb_option.endswith('ckpt'):
        logger.info(f"Loading context embs from ckpt {args.init_context_emb}.")
        state_dict = torch.load(emb_option, map_location=torch.device('cpu'))['state_dict']
        ent_embs = state_dict['ent_embs.weight'].cpu().numpy()
        word_embs = state_dict['word_embs.weight'].cpu().numpy()
        W = state_dict['combine_linear.weight'].cpu().numpy()
        b = state_dict['combine_linear.bias'].cpu().numpy()
    else:
        logger.error(f'init_emb {emb_option} option not recognized, exiting....')
        sys.exit(1)
    logger.info(f'Context embeddings loaded, word_embs : {word_embs.shape}, ent_embs : {ent_embs.shape}')

    word_embs[0] = 0
    ent_embs[0] = 0

    return word_embs, ent_embs, W, b


def get_mention_embs( emb_option=None, num_word=None, num_ent=None, mention_word_dim=None, mention_ent_dim=None):

    if emb_option.endswith('ckpt'):
        logger.info(f"Loading mention embs from {emb_option}.....")
        state_dict = torch.load(emb_option, map_location=torch.device('cpu'))['state_dict']
        mention_word_embs = state_dict['state_dict']['mention_embs.weight']
        mention_ent_embs = state_dict['state_dict']['ent_mention_embs.weight']
    elif emb_option == 'random':
        mention_word_embs = normal_initialize(num_word, mention_word_dim)
        mention_ent_embs = normal_initialize(num_ent, mention_ent_dim)
    else:
        logger.error(f'init_emb {emb_option} option not recognized, exiting....')
        sys.exit(1)

    mention_word_embs[0] = 0
    mention_ent_embs[0] = 0

    return mention_word_embs, mention_ent_embs
