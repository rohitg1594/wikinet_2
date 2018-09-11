# Validator class
import numpy as np
import faiss

from os.path import join
import sys

import re

from logging import getLogger

import torch
from torch.nn import DataParallel

from src.utils.utils import reverse_dict, equalize_len
from src.eval.utils import eval_ranking, check_errors
from src.conll.iter_docs import is_dev_doc, is_test_doc, is_training_doc, iter_docs
from src.tokenizer.regexp_tokenizer import RegexpTokenizer

logger = getLogger(__name__)

RE_DOCID = re.compile('^\d+')


class CombinedValidator:
    def __init__(self,
                 gram_tokenizer=None,
                 gram_dict=None,
                 yamada_model=None,
                 data=None,
                 args=None,):

        self.gram_tokenizer = gram_tokenizer
        self.word_tokenizer = RegexpTokenizer()
        self.word_dict = yamada_model['word_dict']
        self.rev_word_dict = reverse_dict(self.word_dict)
        self.ent_dict = yamada_model['ent_dict']
        self.rev_ent_dict = reverse_dict(self.ent_dict)
        self.gram_dict = gram_dict
        self.rev_gram_dict = reverse_dict(self.gram_dict)
        self.data = data
        self.args = args
        self.model_name = self.args.model_name

        # Get entity tokens
        self.ent_gram_indices, self.ent_word_indices = self._get_ent_tokens()

        # Get wiki mention tokens
        (wiki_all_gold,
         wiki_mention_gram_indices_l,
         wiki_mention_word_indices_l,
         wiki_mention_context_indices_l) = self._get_wiki_mention_tokens()

        # Create numpy arrays
        self.wiki_mention_gram_indices = np.vstack(wiki_mention_gram_indices_l).astype(np.int32)
        self.wiki_mention_word_indices = np.vstack(wiki_mention_word_indices_l).astype(np.int32)
        self.wiki_mention_context_indices = np.vstack(wiki_mention_context_indices_l).astype(np.int32)
        self.wiki_all_gold = np.array(wiki_all_gold).astype(np.int32)

        assert self.wiki_all_gold.shape[0] == self.wiki_mention_word_indices.shape[0] == self.wiki_mention_gram_indices.shape[0]
        assert self.wiki_all_gold.shape[0] == self.wiki_mention_context_indices.shape[0]

        # Mask to select wiki mention queries
        self.wiki_mask = np.random.choice(np.arange(len(self.wiki_mention_gram_indices)),
                                          size=self.args.query_size).astype(np.int32)
        # Get conll mention tokens
        (conll_all_gold,
         conll_mention_gram_indices_l,
         conll_mention_word_indices_l,
         conll_mention_context_indices_l) = self._get_conll_mention_tokens()

        # Create numpy arrays
        self.conll_mention_gram_indices = np.vstack(conll_mention_gram_indices_l).astype(np.int32)
        self.conll_mention_word_indices = np.vstack(conll_mention_word_indices_l).astype(np.int32)
        self.conll_mention_context_indices = np.vstack(conll_mention_context_indices_l).astype(np.int32)
        self.conll_all_gold = np.array(conll_all_gold).astype(np.int32)

        assert self.conll_all_gold.shape[0] == self.conll_mention_word_indices.shape[0] == self.conll_mention_gram_indices.shape[0]
        assert self.conll_all_gold.shape[0] == self.conll_mention_context_indices.shape[0]

        # Debug
        if self.args.debug:
            wiki_debug_result = self._get_debug_string(data='wiki', result=False)
            conll_debug_result = self._get_debug_string(data='conll', result=False)

            print("Wikipedia Debug Results")
            print(wiki_debug_result)

            print("ConllDebug Results")
            print(conll_debug_result)

    def _get_ent_tokens(self):
        """Creates numpy arrays containing gram and word token ids for each entity."""

        # Init Tokens
        ent_gram_tokens = np.zeros((len(self.ent_dict) + 1, self.args.max_gram_size)).astype(np.int32)
        ent_word_tokens = np.zeros((len(self.ent_dict) + 1, self.args.max_word_size)).astype(np.int32)

        # For each entity
        for ent_str, ent_id in self.ent_dict.items():

            # Remove underscore
            ent_str = ent_str.replace('_', ' ')

            # Gram tokens
            gram_tokens = self.gram_tokenizer(ent_str)
            gram_indices = [self.gram_dict.get(token, 0) for token in gram_tokens]
            gram_indices = equalize_len(gram_indices, self.args.max_gram_size)
            ent_gram_tokens[ent_id] = gram_indices

            # Word tokens
            word_tokens = [token.lower() for token in self.word_tokenizer.tokenize(ent_str)]
            word_indices = [self.word_dict.get(token, 0) for token in word_tokens]
            word_indices = equalize_len(word_indices, self.args.max_word_size)
            ent_word_tokens[ent_id] = word_indices

        return ent_gram_tokens, ent_word_tokens

    def _get_wiki_mention_tokens(self):
        """Function for wikipedia data. Creates list of numpy arrays containing gram and word token ids
           for each mention and word tokens for context in abstract. Also output gold entity labels."""

        # Init lists
        all_mention_gram_indices = []
        all_mention_word_indices = []
        all_context_word_indices = []
        all_gold = []

        # For each abstract
        for context_word_tokens, mentions in self.data:

            # For each mention
            for mention, ent_str in mentions:

                # Check if entity is relevant
                if ent_str in self.ent_dict:
                    ent_id = self.ent_dict[ent_str]
                else:
                    continue

                # Gold
                all_gold.append(ent_id)

                # Mention Gram
                mention_gram_tokens = [token for token in self.gram_tokenizer(mention)]
                mention_gram_indices = [self.gram_dict.get(token, 0) for token in mention_gram_tokens]
                mention_gram_indices = equalize_len(mention_gram_indices, self.args.max_gram_size)
                all_mention_gram_indices.append(np.array(mention_gram_indices).astype(np.int64))

                # Mention Word
                mention_word_tokens = [token.lower() for token in self.word_tokenizer.tokenize(mention)]
                mention_word_indices = [self.word_dict.get(token, 0) for token in mention_word_tokens]
                mention_word_indices = equalize_len(mention_word_indices, self.args.max_word_size)
                all_mention_word_indices.append(np.array(mention_word_indices).astype(np.int64))

                # Context Word
                context_word_indices = [self.word_dict.get(token, 0) for token in context_word_tokens]
                context_word_indices = equalize_len(context_word_indices, self.args.max_context_size)
                all_context_word_indices.append(np.array(context_word_indices).astype(np.int64))

        return all_gold, all_mention_gram_indices, all_mention_word_indices, all_context_word_indices

    def _get_conll_mention_tokens(self):
        """Function for CONLL data. Creates list of numpy arrays containing gram and word token ids
           for each mention and word tokens for context in abstract. Also output gold entity labels."""

        # Init lists
        all_mention_gram_indices = []
        all_mention_word_indices = []
        all_context_word_indices = []
        all_gold = []

        # train / dev / test split
        if self.args.conll_split == 'train':
            func = is_training_doc
        elif self.args.conll_split == 'dev':
            func = is_dev_doc
        elif self.args.conll_split == 'test':
            func = is_test_doc
        else:
            logger.error("Conll split {} not recognized, choose one of train, dev, test".format(self.args.conll_split))
            sys.exit(1)

        # For each doc
        for text, gold_ents, _, _, _ in iter_docs(join(self.args.data_path, 'Conll', 'AIDA-YAGO2-dataset.tsv'), func):

            # Context
            context_word_tokens = [token.lower() for token in self.word_tokenizer.tokenize(text)]
            context_word_indices = [self.word_dict.get(token, 0) for token in context_word_tokens]
            context_word_indices = equalize_len(context_word_indices, self.args.max_context_size)

            # For each mention
            for ent_str, (begin, end) in gold_ents:
                if ent_str in self.ent_dict:
                    mention = text[begin:end]
                    all_gold.append(self.ent_dict[ent_str])
                    all_context_word_indices.append(context_word_indices)

                    # Mention Gram
                    mention_gram_tokens = [token for token in self.gram_tokenizer(mention)]
                    mention_gram_indices = [self.gram_dict.get(token, 0) for token in mention_gram_tokens]
                    mention_gram_indices = equalize_len(mention_gram_indices, self.args.max_gram_size)
                    all_mention_gram_indices.append(np.array(mention_gram_indices).astype(np.int64))

                    # Mention Word
                    mention_word_tokens = [token.lower() for token in self.word_tokenizer.tokenize(mention)]
                    mention_word_indices = [self.word_dict.get(token, 0) for token in mention_word_tokens]
                    mention_word_indices = equalize_len(mention_word_indices, self.args.max_word_size)
                    all_mention_word_indices.append(np.array(mention_word_indices).astype(np.int64))

        return all_gold, all_mention_gram_indices, all_mention_word_indices, all_context_word_indices

    def _get_data(self, data_type='wiki'):

        ent_gram_tokens = torch.from_numpy(self.ent_gram_indices).long()
        ent_indices = torch.arange(0, len(self.ent_dict) + 1).long()

        if data_type == 'wiki':
            gram_indices = torch.from_numpy(self.wiki_mention_gram_indices[self.wiki_mask, :]).long()
            word_indices = torch.from_numpy(self.wiki_mention_word_indices[self.wiki_mask, :]).long()
            context_indices = torch.from_numpy(self.wiki_mention_context_indices[self.wiki_mask, :]).long()
        elif data_type == 'conll':
            gram_indices = torch.from_numpy(self.conll_mention_gram_indices).long()
            word_indices = torch.from_numpy(self.conll_mention_word_indices).long()
            context_indices = torch.from_numpy(self.conll_mention_context_indices).long()
        else:
            logger.error('Dataset {} not implemented, choose between wiki and conll'.format(data_type))
            sys.exit(1)

        if self.model_name in ['include_gram', 'weigh_concat']:
            data = (gram_indices, context_indices, ent_gram_tokens, ent_indices)
        elif self.model_name == 'mention prior':
            data = (gram_indices, word_indices, context_indices, ent_gram_tokens, ent_indices)
        elif self.model_name in ['only_prior', 'only_prior_linear']:
            data = (word_indices, ent_indices)
        else:
            logger.error('model {} not implemented'.format(data_type))
            sys.exit(1)

        return data

    def _get_debug_string(self, I=None, data='wiki', result=False):

        if data == 'wiki':
            gram_indices = self.wiki_mention_gram_indices[self.wiki_mask, :]
            word_indices = self.wiki_mention_word_indices[self.wiki_mask, :]
            context_indices = self.wiki_mention_context_indices[self.wiki_mask, :]
            gold = self.wiki_all_gold[self.wiki_mask]
        elif data == 'conll':
            gram_indices = self.conll_mention_gram_indices
            word_indices = self.conll_mention_word_indices
            context_indices = self.conll_mention_context_indices
            gold = self.conll_all_gold
        else:
            logger.error('Dataset {} not implemented, choose between wiki and conll'.format(data))
            sys.exit(1)

        s = ''
        for i in range(10):
            if self.args.include_gram:
                m_g = gram_indices[i]
                s += ''.join([self.rev_gram_dict[token][0] for token in m_g if token in self.rev_gram_dict]) + '|'
            if self.args.include_word:
                m_w = word_indices[i]
                s += ' '.join([self.rev_word_dict[token] for token in m_w if token in self.rev_word_dict]) + '|'
            if self.args.include_context:
                c_w = context_indices[i][:20]
                s += ' '.join([self.rev_word_dict[token] for token in c_w if token in self.rev_word_dict]) + '|'
            s += self.rev_ent_dict[gold[i]] + '|'
            if result:
                s += ','.join([self.rev_ent_dict[ent_id] for ent_id in I[i][:10] if ent_id in self.rev_ent_dict])
            s += '\n'

        return s

    def validate(self, model=None, error=True):
        model = model.eval()
        model = model.cpu()

        input_wiki = self._get_data(data_type='wiki')
        _, wiki_mention_combined_embs = model(input_wiki)
        input_conll = self._get_data(data_type='conll')
        ent_combined_embs, conll_mention_combined_embs = model(input_conll)

        ent_combined_embs = ent_combined_embs.detach().numpy()
        wiki_mention_combined_embs = wiki_mention_combined_embs.detach().numpy()
        conll_mention_combined_embs = conll_mention_combined_embs.detach().numpy()

        # Create / search in Faiss Index
        if self.args.measure == 'ip':
            index = faiss.IndexFlatIP(ent_combined_embs.shape[1])
            logger.info("Using IndexFlatIP")
        else:
            index = faiss.IndexFlatL2(ent_combined_embs.shape[1])
            logger.info("Using IndexFlatL2")
        index.add(ent_combined_embs)

        D_wiki, I_wiki = index.search(wiki_mention_combined_embs.astype(np.float32), 100)
        D_conll, I_conll = index.search(conll_mention_combined_embs.astype(np.float32), 100)

        # Evaluate rankings
        top1_wiki, top10_wiki, top100_wiki, mrr_wiki = eval_ranking(I_wiki, self.wiki_all_gold[self.wiki_mask], [1, 10, 100])
        top1_conll, top10_conll, top100_conll, mrr_conll = eval_ranking(I_conll, self.conll_all_gold, [1, 10, 100])

        # Error analysis
        if error:
            print('Wiki Errors')
            check_errors(I_wiki, self.wiki_all_gold[self.wiki_mask], self.wiki_mention_gram_indices[self.wiki_mask, :],
                         self.rev_ent_dict, self.rev_gram_dict, [1, 10, 100])
            print('\n\n\n')
            print('Conll Errors')
            check_errors(I_conll, self.conll_all_gold, self.conll_mention_gram_indices,
                         self.rev_ent_dict, self.rev_gram_dict, [1, 10, 100])

        if self.args.use_cuda:
            if isinstance(self.args.device, tuple):
                model = model.cuda(self.args.device[0])
                model = DataParallel(model, self.args.device)
            else:
                model = model.cuda(self.args.device)

        return top1_wiki, top10_wiki, top100_wiki, mrr_wiki, top1_conll, top10_conll, top100_conll, mrr_conll
