# Dataloader, based on args.include_word, can include and exclude word information
from os.path import join
import pickle
from logging import getLogger
import sys

import numpy as np
import torch
import torch.utils.data

from src.utils.utils import *
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
        self.word_tokenizer = RegexpTokenizer(lower=True)
        self.model_name = self.args.model_name

        autoencoder_data = pickle_load(join(self.args.data_path, 'autoencoder/data.pickle'))
        self.char_dict = autoencoder_data['char_dict']
        self.max_char_size = self.args.max_char_size

        self.redirects = pickle_load(join(self.args.data_path, 'redirects.pickle'))
        self.num_cand_gen = self.args.num_candidates // 2
        id2context, examples = data
        self.examples = examples
        self.id2context = id2context

        processed_f_name = join(self.args.data_path, 'cache', f'processed_id2context_{self.args.data_type}')
        if os.path.exists(processed_f_name):
            self.processed_id2context = pickle_load(processed_f_name)
        else:
            self.processed_id2context = {}
            for index in self.id2context.keys():
                self.processed_id2context[index] = self._init_context(index)
            with open(processed_f_name, 'wb') as f:
                pickle.dump(self.processed_id2context, f)

        # Candidates
        if not self.args.cand_gen_rand:
            logger.info(f'Loading necounts candidate generation dict.....')
            self.necounts = pickle_load(join(self.args.data_path, 'necounts', 'normal_necounts.pickle'))
            logger.info('necounts loaded.')

    def _get_candidates(self, ent_str, mention):
        """Candidate generation step, can be random or based on necounts."""

        ent_str = self.redirects.get(ent_str, ent_str)
        ent_id = self.ent2id.get(ent_str, 0)
        if self.args.cand_gen_rand:
            candidate_ids = np.concatenate((np.array(ent_id)[None],
                                            np.random.randint(1, self.len_ent + 1,
                                                              size=self.args.num_candidates - 1))).astype(np.int64)
        else:
            nfs = get_normalised_forms(mention)
            candidate_ids = [c_id for nf in nfs for c_id in self.necounts.get(nf, []) if c_id != ent_id]

            if len(candidate_ids) > self.num_cand_gen:
                cand_generation = np.random.choice(np.array(candidate_ids), replace=False, size=self.num_cand_gen)
                num_rand = self.args.num_candidates - self.num_cand_gen - 1
            else:
                cand_generation = np.array(candidate_ids)
                num_rand = self.args.num_candidates - len(candidate_ids) - 1

            cand_random = np.random.randint(1, self.len_ent + 1, num_rand)
            candidate_ids = np.concatenate((np.array(ent_id)[None], cand_generation, cand_random)).astype(np.int64)

        return candidate_ids

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

    def _get_char_tokens(self, s):
        """return char token ids based on self.char_dict for string s."""

        char_ids = [self.char_dict[char] for char in list(s)]
        char_ids = equalize_len_w_eot(char_ids, self.max_char_size, self.char_dict['EOT'])

        return char_ids

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

    def _getitem_only_prior_word_or_gram(self, example, token_type='word'):
        """getitem for only prior and only prior linear models with word or gram tokens."""

        mention, ent_str, span, small_context = example
        candidate_ids = self._get_candidates(ent_str, mention)
        mention_tokens = self._get_tokens(mention, flag=token_type)

        output = {f'mention_{token_type}_tokens': mention_tokens,
                   'candidate_ids': candidate_ids}

        return output


    def _getitem_small_context(self, example):
        """getitem for prior with small context window."""

        mention, ent_str, span, small_context = example
        mention_word_tokens = self._get_tokens(mention, flag='word')
        candidate_ids = self._get_candidates(ent_str, mention)

        output = {'mention_word_tokens': mention_word_tokens,
                  'candidate_ids': candidate_ids,
                  'small_context': small_context}

        return output

    def _getitem_full_context(self, context_id, example):
        """getitem for full context model"""

        mention, ent_str, span, small_context = example
        context_tokens = self.processed_id2context[context_id]
        mention_word_tokens = self._get_tokens(mention, flag='word')
        candidate_ids = self._get_candidates(ent_str, mention)

        output = {'mention_word_tokens': mention_word_tokens,
                  'candidate_ids': candidate_ids,
                  'context_tokens': context_tokens}

        return output

    def _getitem_full_context_string(self, context_id, example):
        """getitem for full context string model"""

        mention, ent_str, span, small_context = example
        context_tokens = self.processed_id2context[context_id]
        mention_word_tokens = self._get_tokens(mention, flag='word')
        candidate_ids = self._get_candidates(ent_str, mention)
        mention_char_tokens = self._get_char_tokens(mention)
        cand_strs = [self.id2ent.get(cand_id, '') for cand_id in candidate_ids]
        candidate_char_tokens = np.vstack([self._get_char_tokens(cand_str) for cand_str in cand_strs])

        output = {'mention_word_tokens': mention_word_tokens,
                  'mention_char_tokens': mention_char_tokens,
                  'context_tokens': context_tokens,
                  'candidate_ids': candidate_ids,
                  'candidate_char_tokens': candidate_char_tokens}

        return output

    def _getitem_pre_train(self, context_id, example):
        """getitem for pre train model."""

        mention, ent_str, span, small_context = example
        context_tokens = self.processed_id2context[context_id]
        candidate_ids = self._get_candidates(ent_str, mention)

        output = {'context_tokens': context_tokens,
                  'candidate_ids': candidate_ids}

        return output

    def __getitem__(self, index):
        """Main getitem function, this calls other getitems based on model type params in self.args."""

        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        # Context Word Tokens
        context_id, example = self.examples[index]

        if self.model_name in ['average', 'linear', 'multi_linear', 'rnn']:
            return self._getitem_only_prior_word_or_gram(example, token_type='word')
        elif self.model_name == 'only_prior_position':
            return self._getitem_only_prior_word_or_gram(example, token_type='word')
        elif self.model_name == 'only_prior_conv':
            return self._getitem_only_prior_word_or_gram(example, token_type='gram')
        elif self.model_name == 'small_context':
            return self._getitem_small_context(example)
        elif self.model_name == 'full_context':
            return self._getitem_full_context(context_id, example)
        elif self.model_name.startswith('full_context_string'):
            print(f'-----------------USING GETITEM FOR FULL CONTEXT STRING----------------{self.model_name}')
            return self._getitem_full_context_string(context_id, example)
        elif self.model_name == 'full_context_attention':
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
