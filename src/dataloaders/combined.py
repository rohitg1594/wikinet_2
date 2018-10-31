# Dataloader, based on args.include_word, can include and exclude word information
from os.path import join
import pickle
from logging import getLogger
import sys

import numpy as np
import torch
import torch.utils.data

from src.utils.utils import reverse_dict, equalize_len, get_normalised_forms, get_absolute_pos
from src.tokenizer.regexp_tokenizer import RegexpTokenizer

logger = getLogger(__name__)


class CombinedDataSet(object):

    def __init__(self,
                 gram_dict=None,
                 word_dict=None,
                 gram_tokenizer=None,
                 ent_dict=None,
                 data=None,
                 args=None):
        """
        data_path: directory path of training data files
        num_shards: number of parts that training file is divided into
        """
        super().__init__()

        self.logger = getLogger(__name__)

        self.ent2id = ent_dict
        self.len_ent = len(self.ent2id)
        self.id2ent = reverse_dict(self.ent2id)

        self.gram_dict = gram_dict
        self.word_dict = word_dict
        self.id2word = reverse_dict(self.word_dict)
        self.gram_tokenizer = gram_tokenizer
        self.args = args
        self.word_tokenizer = RegexpTokenizer(lower=self.args.gram_lower)
        self.model_name = self.args.model_name

        self.num_cand_gen = self.args.num_candidates // 2
        id2context, examples = data
        self.examples = examples
        self.id2context = id2context
        self.logger.info(f'len examples: {len(examples)}, len id2contex: {len(id2context)}')

        self.processed_id2context = {}
        for index in self.id2context.keys():
            self.processed_id2context[index] = self._init_context(index)

        # Candidates
        if not self.args.cand_gen_rand:
            logger.info(f'Loading necounts candidate generation dict.....')
            with open(join(self.args.data_path, 'necounts', 'normal_necounts.pickle'), 'rb') as f:
                self.necounts = pickle.load(f)
            logger.info('necounts loaded.')

    def _get_candidates(self, ent_str, mention, prior=False):
        """Candidate generation step, can be random or based on necounts."""

        ent_id = self.ent2id.get(ent_str, 0)
        if self.args.cand_gen_rand:
            candidate_ids = np.concatenate((np.array(ent_id)[None],
                                            np.random.randint(1, self.len_ent + 1, size=self.args.num_candidates - 1))).astype(np.int64)
        else:
            nfs = get_normalised_forms(mention)
            candidate_ids = []
            for nf in nfs:
                if nf in self.necounts:
                    candidate_ids.extend(self.necounts[nf])

            if ent_id in candidate_ids: candidate_ids.remove(ent_id)  # Remove if true entity is part of candidates

            if len(candidate_ids) > self.num_cand_gen:
                cand_generation = np.random.choice(np.array(candidate_ids), replace=False, size=self.num_cand_gen)
                cand_random = np.random.randint(1, self.len_ent + 1, self.args.num_candidates - self.num_cand_gen - 1)
            else:
                cand_generation = np.array(candidate_ids)
                cand_random = np.random.randint(1, self.len_ent + 1, self.args.num_candidates - len(candidate_ids) - 1)

            candidate_ids = np.concatenate((np.array(ent_id)[None], cand_generation, cand_random)).astype(np.int64)

        if prior:
            priors = np.zeros(self.args.num_candidates)
            nfs = get_normalised_forms(mention)
            for i, cand_id in enumerate(candidate_ids):
                for nf in nfs:
                    if nf in self.necounts:
                        if cand_id in self.necounts[nf]:
                            priors[i] = self.necounts[nf]
                            break
        if not prior:
            return candidate_ids
        else:
            return candidate_ids, priors

    def _init_context(self, index):
        """Initialize numpy array that will hold all context word tokens. Also return mentions"""

        context = self.id2context[index]
        if self.args.ignore_init:
            context = context[5:]
        if len(context) > 0:
            if isinstance(context[0], str):
                context = [self.word_dict.get(token, 0) for token in context]
        context = np.array(equalize_len(context, self.args.max_context_size))

        return context

    def _init_tokens(self, flag='gram'):
        """Initialize numpy array that will hold all mention gram and candidate gram tokens."""

        if flag == 'gram':
            max_size = self.args.max_gram_size
        elif flag == 'word':
            max_size = self.args.max_word_size
        else:
            self.logger.error("flag {} not recognized, choose one of (gram, word)".format(flag))
            sys.exit(1)

        cand_tokens = np.zeros((self.args.num_candidates, max_size)).astype(np.int64)
        mention_tokens = np.zeros(max_size).astype(np.int64)

        return cand_tokens, mention_tokens

    def _get_tokens(self, mention, flag='gram'):
        """Tokenize mention based on flag and then pad them."""

        if flag == 'gram':
            tokens = [self.gram_dict.get(token, 0) for token in self.gram_tokenizer(mention)]
            max_size = self.args.max_gram_size
        elif flag == 'word':
            tokens = [self.word_dict.get(token.text, 0) for token in self.word_tokenizer.tokenize(mention)]
            max_size = self.args.max_word_size
        else:
            self.logger.error("flag {} not recognized, choose one of (gram, word)".format(flag))
            sys.exit(1)

        pad_tokens = np.array(equalize_len(tokens, max_size), dtype=np.int64)

        return pad_tokens

    def _getitem_only_prior_word_or_gram(self, example, token_type='word', include_pos=False):
        """getitem for only prior and only prior linear models with word or gram tokens."""

        _, mention_tokens = self._init_tokens(flag=token_type)
        mention, ent_str, _, _ = example
        cand_ids = np.zeros(self.args.num_candidates).astype(np.int64)

        if ent_str in self.ent2id:
            mention_tokens = self._get_tokens(mention)
            ent_id = self.ent2id[ent_str]
            cand_ids = self._get_candidates(ent_id, mention)

        if include_pos:
            mention_pos = get_absolute_pos(mention_tokens)
            return mention_tokens, mention_pos, cand_ids

        return mention_tokens, cand_ids

    def _getitem_only_prior_word_and_gram(self, example):
        """getitem for only prior and only prior linear models with word and gram tokens."""

        cand_gram_tokens, mention_gram_tokens = self._init_tokens(flag='gram')
        _, mention_word_tokens = self._init_tokens(flag='word')
        cand_ids = np.zeros(self.args.num_candidates).astype(np.int64)

        mention, ent_str, _, _ = example
        if ent_str in self.ent2id:
            ent_id = self.ent2id[ent_str]

            mention_gram_tokens = self._get_tokens(mention, flag='gram')
            mention_word_tokens = self._get_tokens(mention, flag='word')

            cand_ids = self._get_candidates(ent_id, mention)
            cand_strs = [self.id2ent.get(candidate_id, '').replace('_', ' ') for candidate_id in cand_ids]
            cand_gram_tokens = np.array([self._get_tokens(candidate_str, flag='gram') for candidate_str in cand_strs])

        return mention_word_tokens, mention_gram_tokens, cand_gram_tokens, cand_ids

    def _getitem_small_context(self, example):
        """getitem for prior with small context window."""

        _, mention_word_tokens = self._init_tokens(flag='word')
        mention, ent_str, span, small_context = example
        cand_ids = np.zeros(self.args.num_candidates).astype(np.int64)

        if ent_str in self.ent2id:
            ent_id = self.ent2id[ent_str]
            mention_word_tokens = self._get_tokens(mention, flag='word')
            cand_ids = self._get_candidates(ent_id, mention)

        return mention_word_tokens.astype(np.int64), cand_ids.astype(np.int64), small_context.astype(np.int64)

    def _getitem_full_context(self, context_id, example):
        """getitem for prior with small context window."""

        mention, ent_str, span, small_context = example
        context_tokens = self.processed_id2context[context_id]
        mention_word_tokens = self._get_tokens(mention, flag='word')
        cand_ids = self._get_candidates(ent_str, mention)

        return mention_word_tokens, cand_ids, context_tokens

    def _getitem_pre_train(self, context_ids, example):
        """getitem for pre train model."""

        mention, ent_str, span, small_context = example
        cand_ids = np.zeros(self.args.num_candidates).astype(np.int64)

        if ent_str in self.ent2id:
            ent_id = self.ent2id[ent_str]
            cand_ids = self._get_candidates(ent_id, mention)

        return context_ids, cand_ids

    def __getitem__(self, index):
        """Main getitem function, this calls other getitems based on model type params in self.args."""

        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        # Context Word Tokens
        context_id, example = self.examples[index]

        if self.model_name in ['only_prior', 'only_prior_linear', 'only_prior_multi_linear', 'only_prior_rnn']:
            return self._getitem_only_prior_word_or_gram(example, token_type='word', include_pos=False)
        elif self.model_name == 'only_prior_position':
            return self._getitem_only_prior_word_or_gram(example, token_type='word', include_pos=True)
        elif self.model_name == 'only_prior_conv':
            return self._getitem_only_prior_word_or_gram(example, token_type='gram', include_pos=False)
        elif self.model_name == 'only_prior_with_string':
            return self._getitem_only_prior_word_and_gram(example)
        elif self.model_name == 'small_context':
            return self._getitem_small_context(example)
        elif self.model_name == 'full_context':
            return self._getitem_full_context(context_id, example)
        elif self.model_name == 'pre_train':
            return self._getitem_pre_train(context_id, example)
        else:
            self.logger.info('model name {} dataloader not implemented'.format(self.model_name))
            sys.exit(1)

    def __len__(self):
        return len(self.examples)

    def get_loader(self,
                   batch_size=1,
                   shuffle=False,
                   sampler=None,
                   pin_memory=True,
                   drop_last=False,
                   num_workers=4
                   ):

        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last)
