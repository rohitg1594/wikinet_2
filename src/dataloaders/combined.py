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

        self.ent2id = ent_dict
        self.len_ent = len(self.ent2id)
        self.id2ent = reverse_dict(self.ent2id)

        self.gram_dict = gram_dict
        self.word_dict = word_dict
        self.gram_tokenizer = gram_tokenizer
        self.args = args
        self.word_tokenizer = RegexpTokenizer(lower=self.args.gram_lower)
        self.model_name = self.args.model_name

        self.candidate_generation = self.args.num_candidates // 2
        self.data = data
        self.logger = getLogger(__name__)

        # Candidates
        if not self.args.cand_gen_rand:
            logger.info(f'Loading necounts candidate generation dict.....')
            with open(join(self.args.data_path, 'necounts', 'normal_necounts.pickle'), 'rb') as f:
                self.necounts = pickle.load(f)
            logger.info('necounts loaded.')

    def _get_candidates(self, ent_id, mention, prior=False):
        """Candidate generation step, can be random or based on necounts."""

        if self.args.cand_gen_rand:
            candidate_ids = np.concatenate((np.array(ent_id)[None],
                                            np.random.randint(1, self.len_ent + 1, size=self.args.num_candidates - 1)))
        else:
            nfs = get_normalised_forms(mention)
            candidates = []
            for nf in nfs:
                if nf in self.necounts:
                    candidates.extend(self.necounts[nf])

            candidate_ids = [self.ent2id[candidate] for candidate in candidates if candidate in self.ent2id]

            if ent_id in candidate_ids: candidate_ids.remove(ent_id)  # Remove if true entity is part of candidates

            if len(candidate_ids) > self.candidate_generation:
                cand_generation = np.random.choice(np.array(candidate_ids), replace=False, size=self.candidate_generation)
                cand_random = np.random.randint(1, self.len_ent + 1, self.args.num_candidates - self.candidate_generation - 1)
            else:
                cand_generation = np.array(candidate_ids)
                cand_random = np.random.randint(1, self.len_ent + 1, self.args.num_candidates - len(candidate_ids) - 1)

            candidate_ids = np.concatenate((np.array(ent_id)[None], cand_generation, cand_random))

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

        context_word_tokens, examples = self.data[index]
        context_word_tokens = [self.word_dict.get(token, 0) for token in context_word_tokens]
        context_word_tokens = np.array(equalize_len(context_word_tokens, self.args.max_context_size))

        # TODO - maybe this is too expensive
        context_word_tokens_array = np.zeros((self.args.max_ent_size, self.args.max_context_size), dtype=np.int64)
        context_word_tokens_array[:len(examples)] = context_word_tokens

        return context_word_tokens_array, examples

    def _init_tokens(self, flag='gram'):
        """Initialize numpy array that will hold all mention gram and candidate gram tokens."""

        if flag == 'gram':
            max_size = self.args.max_gram_size
        elif flag == 'word':
            max_size = self.args.max_word_size
        else:
            self.logger.error("flag {} not recognized, choose one of (gram, word)".format(flag))
            sys.exit(1)

        cand_tokens = np.zeros((self.args.max_ent_size, self.args.num_candidates, max_size)).astype(np.int64)
        mention_tokens = np.zeros((self.args.max_ent_size, max_size)).astype(np.int64)

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

    def _getitem_only_prior_word_or_gram(self, mask, examples, all_candidate_ids, token_type='word', include_pos=False):
        """getitem for only prior and only prior linear models with word or gram tokens."""

        _, all_mention_tokens = self._init_tokens(flag=token_type)

        for ent_idx, (mention, ent_str, _, _) in enumerate(examples[:self.args.max_ent_size]):
            if ent_str in self.ent2id:
                ent_id = self.ent2id[ent_str]
            else:
                continue

            # Mention Tokens
            all_mention_tokens[ent_idx] = self._get_tokens(mention, flag=token_type)

            # Candidate Generation
            candidate_ids = self._get_candidates(ent_id, mention)
            all_candidate_ids[ent_idx] = candidate_ids

        if include_pos:
            all_mention_pos = get_absolute_pos(all_mention_tokens)
            return mask, all_mention_tokens, all_mention_pos, all_candidate_ids

        return mask, all_mention_tokens, all_candidate_ids

    def _getitem_only_prior_word_and_gram(self, mask, examples, all_candidate_ids):
        """getitem for only prior and only prior linear models with word and gram tokens."""

        all_candidate_gram_tokens, all_mention_gram_tokens = self._init_tokens(flag='gram')
        _, all_mention_word_tokens = self._init_tokens(flag='word')

        for ent_idx, (mention, ent_str, _, _) in enumerate(examples[:self.args.max_ent_size]):
            if ent_str in self.ent2id:
                ent_id = self.ent2id[ent_str]
            else:
                continue

            # Mention Tokens
            all_mention_gram_tokens[ent_idx] = self._get_tokens(mention, flag='gram')
            all_mention_word_tokens[ent_idx] = self._get_tokens(mention, flag='word')

            # Candidate Generation
            candidate_ids = self._get_candidates(ent_id, mention)
            all_candidate_ids[ent_idx] = candidate_ids

            # Candidate Gram Tokens
            candidate_strs = [self.id2ent.get(candidate_id, '').replace('_', ' ') for candidate_id in candidate_ids]
            candidate_gram_tokens = np.array([self._get_tokens(candidate_str, flag='gram') for candidate_str in candidate_strs])

            all_candidate_gram_tokens[ent_idx] = candidate_gram_tokens

        return mask, all_mention_word_tokens, all_mention_gram_tokens, all_candidate_gram_tokens, all_candidate_ids

    def _getitem_small_context(self, mask, examples, all_candidate_ids, window=5):
        """getitem for prior with small context window."""

        _, all_mention_word_tokens = self._init_tokens(flag='word')
        all_small_context = np.zeros((self.args.max_ent_size, 2 * window), dtype=np.int64)

        for ent_idx, (mention, ent_str, span, small_context) in enumerate(examples[:self.args.max_ent_size]):
            if ent_str in self.ent2id:
                ent_id = self.ent2id[ent_str]
            else:
                continue

            # Mention Tokens
            all_mention_word_tokens[ent_idx] = self._get_tokens(mention, flag='word')

            # Candidate Generation
            candidate_ids = self._get_candidates(ent_id, mention)
            all_candidate_ids[ent_idx] = candidate_ids

            # Context
            all_small_context[ent_idx] = small_context

        return mask, all_mention_word_tokens, all_candidate_ids, all_small_context

    def _getitem_only_prior_regress(self, mask, examples, all_candidate_ids):
        """getitem for only prior with regression model."""

        all_candidate_words, all_mention_words = self._init_tokens(flag='word')
        all_candidate_priors = np.zeros_like(all_candidate_ids)

        for ent_idx, (mention, ent_str, _, _) in enumerate(examples[:self.args.max_ent_size]):
            if ent_str in self.ent2id:
                ent_id = self.ent2id[ent_str]
            else:
                continue

            # Mention Word Tokens
            all_mention_words[ent_idx] = self._get_tokens(mention, flag='word')

            # Candidate Generation
            candidate_ids, priors = self._get_candidates(ent_id, mention)
            all_candidate_ids[ent_idx] = candidate_ids
            all_candidate_priors[ent_idx] = priors

        return mask, all_mention_words, all_candidate_ids, all_candidate_priors

    def _getitem_only_prior_full(self, example):

        assert len(example) == 1
        mention, ent_str = example[0]
        mention_word_tokens = self._get_tokens(mention, flag='word')
        if ent_str in self.ent2id:
            mask = np.array([1], dtype=np.float32)
            label = self.ent2id[ent_str]
        else:
            mask = np.array([0], dtype=np.float32)
            label = 0

        return mask, label, mention_word_tokens

    def _getitem_include_word(self, mask, examples, all_candidate_ids, all_context_words):
        """getitem for model which include mention and candidate words."""

        # Init Grams
        all_candidate_grams, all_mention_grams = self._init_tokens(flag='gram')

        # Init Words
        all_candidate_words, all_mention_words = self._init_tokens(flag='word')

        # For each mention
        for ent_idx, (mention, ent_str, _, _) in enumerate(examples[:self.args.max_ent_size]):
            if ent_str in self.ent2id:
                ent_id = self.ent2id[ent_str]
            else:
                continue

            # Mention Gram Tokens
            all_mention_grams[ent_idx] = self._get_tokens(mention, flag='gram')

            # Mention Word Tokens
            all_mention_words[ent_idx] = self._get_tokens(mention, flag='word')

            # Candidate Generation
            candidate_ids = self._get_candidates(ent_id, mention)
            all_candidate_ids[ent_idx] = candidate_ids

            # Gram and word tokens for Candidates
            candidate_gram_tokens = np.zeros((self.args.num_candidates, self.args.max_gram_size)).astype(np.int64)
            candidate_word_tokens = np.zeros((self.args.num_candidates, self.args.max_word_size)).astype(np.int64)

            for cand_idx, candidate_id in enumerate(candidate_ids):
                candidate_str = self.id2ent.get(candidate_id, '').replace('_', ' ')

                # Candidate Gram Tokens
                candidate_gram_tokens[cand_idx] = self._get_tokens(candidate_str, flag='gram')

                # Candidate Word Tokens
                candidate_word_tokens[cand_idx] = self._get_tokens(candidate_str, flag='word')

            all_candidate_grams[ent_idx] = candidate_gram_tokens
            all_candidate_words[ent_idx] = candidate_word_tokens

        return (mask,
                all_mention_grams,
                all_mention_words,
                all_context_words,
                all_candidate_grams,
                all_candidate_words,
                all_candidate_ids)

    def _getitem_mention_prior(self, mask, examples, all_candidate_ids, all_context_words):

        # Init Grams
        all_candidate_grams, all_mention_grams = self._init_tokens(flag='gram')

        # Init Words
        all_candidate_words, all_mention_words = self._init_tokens(flag='word')

        # For each mention
        for ent_idx, (mention, ent_str, _, _) in enumerate(examples[:self.args.max_ent_size]):
            if ent_str in self.ent2id:
                ent_id = self.ent2id[ent_str]
            else:
                continue

            # Mention Gram Tokens
            all_mention_grams[ent_idx] = self._get_tokens(mention, flag='gram')

            # Mention Word Tokens
            all_mention_words[ent_idx] = self._get_tokens(mention, flag='word')

            # Candidate Generation
            candidate_ids = self._get_candidates(ent_id, mention)
            all_candidate_ids[ent_idx] = candidate_ids

            # Gram and word tokens for Candidates
            candidate_gram_tokens = np.zeros((self.args.num_candidates, self.args.max_gram_size)).astype(np.int64)

            for cand_idx, candidate_id in enumerate(candidate_ids):
                candidate_str = self.id2ent.get(candidate_id, '').replace('_', ' ')

                # Candidate Gram Tokens
                candidate_gram_tokens[cand_idx] = self._get_tokens(candidate_str, flag='gram')

            all_candidate_grams[ent_idx] = candidate_gram_tokens

        return (mask,
                all_mention_grams,
                all_mention_words,
                all_context_words,
                all_candidate_grams,
                all_candidate_ids)

    def _getitem_include_gram(self, mask, examples, all_candidate_ids, all_context_words):

        # Init Grams
        all_candidate_grams, all_mention_grams = self._init_tokens(flag='gram')

        for ent_idx, (mention, ent_str, _, _) in enumerate(examples[:self.args.max_ent_size]):
            if ent_str in self.ent2id:
                ent_id = self.ent2id[ent_str]
            else:
                continue

            # Mention Gram Tokens
            all_mention_grams[ent_idx] = self._get_tokens(mention, flag='gram')

            # Candidate Generation
            candidate_ids = self._get_candidates(ent_id, mention)
            all_candidate_ids[ent_idx] = candidate_ids

            # Candidate Gram tokens
            candidate_gram_tokens = np.zeros((self.args.num_candidates, self.args.max_gram_size)).astype(np.int64)

            for cand_idx, candidate_id in enumerate(candidate_ids):
                candidate_str = self.id2ent.get(candidate_id, '').replace('_', ' ')
                candidate_gram_tokens[cand_idx] = self._get_tokens(candidate_str, flag='gram')

            all_candidate_grams[ent_idx] = candidate_gram_tokens

        return mask, all_mention_grams, all_context_words, all_candidate_grams, all_candidate_ids

    def __getitem__(self, index):
        """Main getitem function, this calls other getitems based on model type params in self.args."""

        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        # Init candidate ids
        all_candidate_ids = np.zeros((self.args.max_ent_size, self.args.num_candidates)).astype(np.int64)

        # Context Word Tokens
        all_context_tokens, examples = self._init_context(index)

        # Mask of indices to ignore for final loss
        mask = np.zeros(self.args.max_ent_size, dtype=np.float32)
        mask[:len(examples)] = 1

        if self.model_name in ['only_prior', 'only_prior_linear', 'only_prior_multi_linear', 'only_prior_rnn']:
            return self._getitem_only_prior_word_or_gram(mask, examples, all_candidate_ids, token_type='word', include_pos=False)
        elif self.model_name == 'only_prior_position':
            return self._getitem_only_prior_word_or_gram(mask, examples, all_candidate_ids, token_type='word', include_pos=True)
        elif self.model_name == 'only_prior_conv':
            return self._getitem_only_prior_word_or_gram(mask, examples, all_candidate_ids, token_type='gram', include_pos=False)
        elif self.model_name == 'only_prior_with_string':
            return self._getitem_only_prior_word_and_gram(mask, examples, all_candidate_ids)
        elif self.model_name == 'prior_small_context':
            return self._getitem_small_context(mask, examples, all_candidate_ids, window=self.args.context_window)
        elif self.model_name == 'only_prior_full':
            return self._getitem_only_prior_full(examples)
        elif self.model_name == 'include_word':
            return self._getitem_include_word(mask, examples, all_candidate_ids, all_context_tokens)
        elif self.model_name == 'mention_prior':
            return self._getitem_mention_prior(mask, examples, all_candidate_ids, all_context_tokens)
        elif self.model_name in ['include_gram', 'weigh_concat']:
            return self._getitem_include_gram(mask, examples, all_candidate_ids, all_context_tokens)
        else:
            self.logger.info('model name {} dataloader not implemented'.format(self.model_name))

    def __len__(self):
        return len(self.data)

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



