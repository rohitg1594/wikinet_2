# Various Utility Functions to be used elsewhere
import string
import re
import json
import random
from collections import defaultdict
import os
from os.path import join
import sys
import logging
import gensim
import pickle

import numpy as np
import torch
from torch.nn import DataParallel

# Leave this here to prevent circular imports!
def np_to_tensor(a):
    if isinstance(a, np.ndarray):
        return torch.from_numpy(a)
    else:
        return a

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
    embs[0] = 0

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


def equalize_len(data, max_size, pad=0):
    d = data.copy()
    l = len(d)

    if l >= max_size:
        return d[:max_size]
    else:
        for _ in range(max_size - l):
            d.append(pad)

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


def check_errors(I, gold, gram_indices, rev_ent_dict, rev_gram_dict, redirects, ks):
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
    out = {k: 0 for k in ks}

    for k in ks:
        for i in range(I.shape[0]):
            if gold[i] in I[i, :k]:
                out[k] += 1

    out = {k: v / I.shape[0] for k, v in out.items()}

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

    out['mrr'] = mrr

    return out


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def create_arr(strs, max_char, char_dict, ent2id=None):
    dim_0 = len(strs) + 1 if ent2id else len(strs)
    arr = np.zeros((dim_0, max_char), dtype=np.int64)

    for i, s in enumerate(strs):
        char_ids = [char_dict[char] for char in list(s.replace('_', ' '))]
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


def get_context_embs(data_path=None, emb_option=None, yamada_model=None, ent_emb_init=None):

    num_word, word_dim = yamada_model['word_emb'].shape
    num_ent, ent_dim = yamada_model['ent_emb'].shape

    if emb_option == 'random':
        logger.info(f"Initializing context embs randomly.....")
        word_embs = normal_initialize(num_word, word_dim)
        ent_embs = normal_initialize(num_ent, ent_dim)
        W = normal_initialize(word_dim, ent_dim)
        b = np.random.randn(ent_dim)
    elif 'w2v' in emb_option:
        logger.info(f"Loading context embs from {emb_option}.....")
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
        logger.info(f"Loading context embs from ckpt {emb_option}.")
        state_dict = torch.load(emb_option, map_location=torch.device('cpu'))['state_dict']
        ent_embs = state_dict['ent_embs.weight'].cpu().numpy()
        word_embs = state_dict['word_embs.weight'].cpu().numpy()
        W = state_dict['combine_linear.weight'].cpu().numpy()
        b = state_dict['combine_linear.bias'].cpu().numpy()
    else:
        logger.error(f'init_emb {emb_option} option not recognized, exiting....')
        sys.exit(1)

    word_embs[0] = 0
    ent_embs[0] = 0

    return word_embs, ent_embs, W, b


def get_mention_embs( emb_option=None, num_word=None, num_ent=None, mention_word_dim=None, mention_ent_dim=None):

    if emb_option.endswith('ckpt'):
        logger.info(f"Loading mention embs from {emb_option}.....")
        state_dict = torch.load(emb_option, map_location=torch.device('cpu'))['state_dict']
        mention_word_embs = state_dict['mention_word_embs.weight']
        mention_ent_embs = state_dict['mention_ent_embs.weight']
    elif emb_option == 'random':
        mention_word_embs = normal_initialize(num_word, mention_word_dim)
        mention_ent_embs = normal_initialize(num_ent, mention_ent_dim)
    else:
        logger.error(f'init_emb {emb_option} option not recognized, exiting....')
        sys.exit(1)

    mention_word_embs[0] = 0
    mention_ent_embs[0] = 0

    return mention_word_embs, mention_ent_embs


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


def json_load(path):
    assert os.path.exists(path)

    with open(path, 'r') as f:
        data = json.load(f)

    return data


def pickle_dump(o, path):

    with open(path, 'wb') as f:
        pickle.dump(o, f)


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


def load_data(data_type, train_size, data_path):
    """
       Load train data in format used by combined and yamada dataloader.
    """
    res = {}
    splits = ['train', 'dev', 'test']
    if data_type == 'proto':
        logger.info("Loading Wikipedia proto training data.....")
        for split in ['train', 'dev']:
            id2context, examples = pickle_load(join(data_path, 'training_files', 'proto', f'{split}.pickle'))
            if split == 'train':
                examples = examples[:train_size]
            res[split] = id2context, examples

        res['test'] = {}, []

    elif data_type == 'full':
        logger.info("Loading Wikipedia orig training data.....")
        id2context, examples = pickle_load(join(data_path, 'training_files', 'full', 'full.pickle'))

        train_data = []
        dev_data = []
        test_data = []
        for ex in examples:
            if len(train_data) == train_size:
                break
            r = np.random.random()
            if r < 0.90:
                train_data.append(ex)

            elif 0.9 < r < 0.95:
                dev_data.append(ex)
            else:
                test_data.append(ex)

        res = {'train': (id2context, train_data),
               'dev': (id2context, dev_data),
               'test': (id2context, test_data)
               }
    elif data_type == 'conll':
        data_dict = pickle_load(join(data_path, 'training_files', f'all_conll.pickle'))
        for split in splits:
            res[split] = data_dict['id2c'], data_dict['data'][split]
    else:
        logger.error("Data type {} not recognized".format(data_type))
        sys.exit(1)

    return res


def load_gensim(data_path=None, model_dir=None, yamada_model=None):
    """Load model trained with gensim, fill in the ent and word vector matrix and return them."""
    word2id = yamada_model['word_dict']
    ent2id = yamada_model['ent_dict']

    model = gensim.models.KeyedVectors.load(join(data_path, 'w2v', model_dir, 'model'))
    wv = model.wv
    index2word = wv.index2word
    vectors = wv.vectors
    emb_dim = vectors.shape[1]

    ent_indices = [i for i, word in enumerate(index2word) if word.startswith('e__')]
    ent_ids = [int(word[3:]) for i, word in enumerate(index2word) if word.startswith('e__')]

    word_indices = [i for i, word in enumerate(index2word) if not word.startswith('e__')]
    words = [word for i, word in enumerate(index2word) if not word.startswith('e__')]

    ent_embs = np.zeros((len(ent2id) + 1, emb_dim))
    for ent_index, ent_id in zip(ent_indices, ent_ids):
        ent_embs[ent_id] = vectors[ent_index]

    word_embs = np.zeros((len(word2id) + 1, emb_dim))
    for word_index, word in zip(word_indices, words):
        if word in word2id:
            word_id = word2id[word]
            word_embs[word_id] = vectors[word_index]

    return ent_embs, word_embs
