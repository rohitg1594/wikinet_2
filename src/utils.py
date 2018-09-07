# Various Utility Functions to be used elsewhere
import numpy as np

import string
import re
import sys
import logging

import torch
from torch.nn import DataParallel
import torch.nn as nn

from src.models.combined.include_gram import IncludeGram
from src.models.combined.include_word import IncludeWord
from src.models.combined.mention_prior import MentionPrior
from src.models.combined.weigh_concat import WeighConcat
from src.models.combined.only_prior import OnlyPrior
from src.models.combined.only_prior_linear import OnlyPriorLinear

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


def get_model(args, yamada_model=None, gram_embs=None, ent_embs=None, word_embs=None, init='xavier_normal'):
    """Based on parameters in args, initialize and return appropriate model."""
    kwargs = {'args': args,
              'gram_embs': gram_embs,
              'ent_embs': ent_embs,
              'word_embs': word_embs,
              'W': yamada_model['W'],
              'b': yamada_model['b']}
    model_name = args.model_name

    if model_name == 'include_word':
        model_type = IncludeWord
    elif model_name == 'weigh_concat':
        model_type = WeighConcat
    elif model_name == 'include_gram':
        model_type = IncludeGram
    elif model_name in ['only_prior', 'only_prior_linear', 'mention_prior']:
        mention_embs = torch.Tensor(word_embs.shape[0], args.mention_word_dim)
        ent_mention_embs = torch.Tensor(ent_embs.shape[0], args.mention_word_dim)

        # Initialization
        if init == 'normal':
            mention_embs = torch.from_numpy(normal_initialize(word_embs.shape[0], args.mention_word_dim))
            ent_mention_embs = torch.from_numpy(normal_initialize(ent_embs.shape[0], args.mention_word_dim))
        elif init == 'xavier_uniform':
            nn.init.xavier_uniform(mention_embs)
            nn.init.xavier_uniform(ent_mention_embs)
        elif init == 'xavier_normal':
            nn.init.xavier_normal(mention_embs)
            nn.init.xavier_normal(ent_mention_embs)
        elif init == 'kaiming_uniform':
            nn.init.kaiming_uniform(mention_embs)
            nn.init.kaiming_uniform(ent_mention_embs)
        elif init == 'kaiming_normal':
            nn.init.kaiming_normal(mention_embs)
            nn.init.kaiming_normal(ent_mention_embs)

        mention_embs[0] = 0
        ent_mention_embs[0] = 0
        kwargs['mention_embs'] = mention_embs
        kwargs['ent_mention_embs'] = ent_mention_embs

        if model_name == 'only_prior':
            model_type = OnlyPrior
        elif model_name == 'only_prior_linear':

            # Initialize linear layer with identity matrix
            mention_linear_W = torch.Tensor(mention_embs.shape[1], mention_embs.shape[1])
            torch.nn.init.eye(mention_linear_W)
            mention_linear_b = torch.Tensor(mention_embs.shape[1])
            torch.nn.init.constant(mention_linear_b, 0)

            if args.debug:
                print('Mention Linear W:')
                print(mention_linear_W)

                print('\n\nMention Linear b:')
                print(mention_linear_b)

            kwargs['mention_linear_W'] = mention_linear_W
            kwargs['mention_linear_b'] = mention_linear_b

            model_type = OnlyPriorLinear
        else:
            model_type = MentionPrior
    else:
        logger.error("model name {} not recognized".format(model_name))
        sys.exit(1)

    model = model_type(**kwargs)
    if args.use_cuda:
        if isinstance(args.device, tuple):
            model = model.cuda(args.device[0])
            model = DataParallel(model, args.device)
        else:
            model = model.cuda(args.device)
    logger.info('{} Model created.'.format(model_type.__name__))

    return model
