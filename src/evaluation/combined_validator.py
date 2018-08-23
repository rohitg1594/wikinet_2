# Validator class
import numpy as np
import faiss

from os.path import join
import sys

from collections import OrderedDict
import random
import re

from logging import getLogger

from src.utils import reverse_dict, equalize_len, normalize
from src.evaluation.eval_utils import eval_ranking, check_errors
from src.conll.iter_docs import is_dev_doc, is_test_doc, is_training_doc, iter_docs

logger = getLogger()

RE_DOCID = re.compile('^\d+')


class CombinedValidator:
    def __init__(self,
                 gram_tokenizer=None,
                 gram_dict=None,
                 yamada_model=None,
                 data=None,
                 args=None,):

        self.tokenizer = gram_tokenizer
        self.word_dict = yamada_model['word_dict']
        self.rev_word_dict = reverse_dict(self.word_dict)
        self.ent_dict = yamada_model['ent_dict']
        self.rev_ent_dict = reverse_dict(self.ent_dict)
        self.gram_dict = gram_dict
        self.rev_gram_dict = reverse_dict(self.gram_dict)
        self.data = data
        self.args = args

        self.ent_gram_indices, self.ent_word_indices = self._get_ent_tokens()

        (wiki_all_gold,
         wiki_mention_gram_indices_l,
         wiki_mention_word_indices_l,
         wiki_mention_context_indices_l) = self._get_wiki_mention_tokens()

        self.wiki_mention_gram_indices = np.vstack(wiki_mention_gram_indices_l).astype(np.int32)
        self.wiki_mention_word_indices = np.vstack(wiki_mention_word_indices_l).astype(np.int32)
        self.wiki_mention_context_indices = np.vstack(wiki_mention_context_indices_l).astype(np.int32)
        self.wiki_all_gold = np.array(wiki_all_gold).astype(np.int32)
        self.wiki_mask = np.random.choice(np.arange(len(self.wiki_mention_gram_indices)),
                                          size=self.args.query_size).astype(np.int32)

        (conll_all_gold,
         conll_mention_gram_indices_l,
         conll_mention_word_indices_l,
         conll_mention_context_indices_l) = self._get_conll_mention_tokens()

        self.conll_mention_gram_indices = np.vstack(conll_mention_gram_indices_l).astype(np.int32)
        self.conll_mention_word_indices = np.vstack(conll_mention_word_indices_l).astype(np.int32)
        self.conll_mention_context_indices = np.vstack(conll_mention_context_indices_l).astype(np.int32)
        self.conll_all_gold = np.array(conll_all_gold).astype(np.int32)

        if self.args.debug:
            wiki_debug_result = self._get_debug_string(data='wiki', result=False)
            conll_debug_result = self._get_debug_string(data='conll', result=False)

            print("Wikipedia Debug Results")
            print(wiki_debug_result)

            print("ConllDebug Results")
            print(conll_debug_result)

    def _get_ent_tokens(self):
        ent_gram_tokens = np.zeros((len(self.ent_dict) + 1, self.args.max_gram_size)).astype(np.int32)
        ent_word_tokens = np.zeros((len(self.ent_dict) + 1, self.args.max_word_size)).astype(np.int32)
        for ent_str, ent_id in self.ent_dict.items():
            gram_tokens = self.tokenizer(ent_str.replace('_', ' '))
            gram_indices = [self.gram_dict.get(token, 0) for token in gram_tokens]
            gram_indices = equalize_len(gram_indices, self.args.max_gram_size)
            ent_gram_tokens[ent_id] = gram_indices

            word_tokens = ent_str.replace('_', ' ').lower().split()
            word_indices = [self.word_dict.get(token, 0) for token in word_tokens]
            word_indices = equalize_len(word_indices, self.args.max_word_size)
            ent_word_tokens[ent_id] = word_indices

        return ent_gram_tokens, ent_word_tokens

    def _get_wiki_mention_tokens(self):
        all_mention_gram_tokens = []
        all_mention_word_tokens = []
        all_context_word_tokens = []
        all_gold = []
        for tokens, mentions in self.data:
            for mention, ent_str in mentions:
                if ent_str in self.ent_dict:
                    ent_id = self.ent_dict[ent_str]
                else:
                    continue

                # Gold
                all_gold.append(ent_id)

                # Mention Gram Tokens
                mention_gram_tokens = [self.gram_dict[token] for token in self.tokenizer(mention) if token in self.gram_dict]
                mention_gram_tokens = equalize_len(mention_gram_tokens, self.args.max_gram_size)
                all_mention_gram_tokens.append(np.array(mention_gram_tokens).astype(np.int64))

                # Mention Word Tokens
                mention_word_tokens = [self.word_dict[token] for token in mention.lower().split() if token in self.word_dict]
                mention_word_tokens = equalize_len(mention_word_tokens, self.args.max_word_size)
                all_mention_word_tokens.append(np.array(mention_word_tokens).astype(np.int64))

                # Context Word Tokens
                context_word_tokens = [self.word_dict[token] for token in tokens if token in self.word_dict]
                context_word_tokens = equalize_len(context_word_tokens, self.args.max_context_size)
                all_context_word_tokens.append(np.array(context_word_tokens).astype(np.int64))

        return all_gold, all_mention_gram_tokens, all_mention_word_tokens, all_context_word_tokens

    def _get_conll_mention_tokens(self):
        all_context_word_tokens = []
        all_mention_word_tokens = []
        all_mention_gram_tokens = []
        all_gold = []

        if self.args.conll_split == 'train':
            func = is_training_doc
        elif self.args.conll_split == 'dev':
            func = is_dev_doc
        elif self.args.conll_split == 'test':
            func = is_test_doc
        else:
            logger.error("Conll split {} not recognized, choose one of train, dev, test".format(self.args.conll_split))
            sys.exit(1)

        for text, gold_ents, _, _, _ in iter_docs(join(self.args.data_path, 'Conll', 'AIDA-YAGO2-dataset.tsv'), func):
            context_word_tokens = [self.word_dict.get(token, 0) for token in text.lower().split()]
            context_word_tokens = equalize_len(context_word_tokens, self.args.max_context_size)
            for ent_str, (begin, end) in gold_ents:
                if ent_str in self.ent_dict:
                    mention = text[begin:end]
                    all_gold.append(self.ent_dict[ent_str])
                    all_context_word_tokens.append(context_word_tokens)

                    mention_gram_tokens = [self.gram_dict.get(token, 0) for token in self.tokenizer(mention)]
                    mention_gram_tokens = equalize_len(mention_gram_tokens, self.args.max_gram_size)
                    all_mention_gram_tokens.append(mention_gram_tokens)

                    mention_word_tokens = [self.word_dict.get(token, 0) for token in mention.lower().split()]
                    mention_word_tokens = equalize_len(mention_word_tokens, self.args.max_word_size)
                    all_mention_word_tokens.append(mention_word_tokens)

        return all_gold, all_mention_gram_tokens, all_mention_word_tokens, all_context_word_tokens

    def _get_model_params(self, model):
        params = dict()
        new_state_dict = OrderedDict()
        for k, v in model.state_dict().items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v

        params['word_embs'] = new_state_dict['word_embs.weight'].cpu().numpy()
        params['ent_embs'] = new_state_dict['ent_embs.weight'].cpu().numpy()
        params['gram_embs'] = new_state_dict['gram_embs.weight'].cpu().numpy()
        params['W'] = new_state_dict['orig_linear.weight'].cpu().numpy()
        params['b'] = new_state_dict['orig_linear.bias'].cpu().numpy()

        if self.args.include_mention:
            params['mention_embs'] = new_state_dict['word_embs.weight'].cpu().numpy()
            params['ent_mention_embs'] = new_state_dict['ent_embs.weight'].cpu().numpy()

        return params

    def _get_ent_combined_embs(self, params=None):

        gram_embs = params['gram_embs']
        word_embs = params['word_embs']
        ent_embs = params['ent_embs']
        W = params['W']
        b = params['b']

        if self.args.include_mention:
            ent_mention_embs = params['ent_mention_embs'][self.ent_word_indices, :].mean(axis=1)

            if self.args.norm_mention:
                ent_mention_embs = normalize(ent_mention_embs)

        if self.args.include_gram:
            ent_gram_embs = gram_embs[self.ent_gram_indices, :].mean(axis=1)

            if self.args.norm_gram:
                ent_gram_embs = normalize(ent_gram_embs)

        if self.args.include_word:
            ent_word_embs = word_embs[self.ent_word_indices, :].mean(axis=1)
            ent_word_embs = ent_word_embs @ W + b

            if self.args.norm_word:
                ent_word_embs = normalize(ent_word_embs)

        if self.args.include_mention:
            ent_combined_embs = np.concatenate((ent_embs, ent_gram_embs, ent_mention_embs), axis=1)
        else:
            # 100
            if self.args.include_gram and not self.args.include_context and not self.args.include_word:
                ent_combined_embs = ent_gram_embs
            # 001
            elif not self.args.include_gram and not self.args.include_context and self.args.include_word:
                ent_combined_embs = ent_word_embs
            # 010
            elif not self.args.include_gram and self.args.include_context and not self.args.include_word:
                ent_combined_embs = ent_embs
            # 110
            elif self.args.include_gram and self.args.include_context and not self.args.include_word:
                ent_combined_embs = np.concatenate((ent_embs, ent_gram_embs), axis=1)
            # 101
            elif self.args.include_gram and not self.args.include_context and self.args.include_word:
                ent_combined_embs = np.concatenate((ent_word_embs, ent_gram_embs), axis=1)
            # 011
            elif not self.args.include_gram and self.args.include_context and self.args.include_word:
                ent_combined_embs = np.concatenate((ent_embs, ent_word_embs), axis=1)
            # 111
            else:
                ent_combined_embs = np.concatenate((ent_embs, ent_gram_embs, ent_word_embs), axis=1)

        if self.args.norm_final:
            ent_combined_embs = normalize(ent_combined_embs)

        return ent_combined_embs

    def _get_mention_combined_embs(self, params=None, data='wiki'):
        gram_embs = params['gram_embs']
        word_embs = params['word_embs']
        W = params['W']
        b = params['b']

        if data == 'wiki':
            gram_indices = self.wiki_mention_gram_indices[self.wiki_mask, :]
            word_indices = self.wiki_mention_word_indices[self.wiki_mask, :]
            context_indices = self.wiki_mention_context_indices[self.wiki_mask, :]
        elif data == 'conll':
            gram_indices = self.conll_mention_gram_indices
            word_indices = self.conll_mention_word_indices
            context_indices = self.conll_mention_context_indices
        else:
            logger.error('Dataset {} not implemented, choose between wiki and conll'.format(data))
            sys.exit(1)

        if self.args.include_mention:
            mention_embs = params['mention_embs'][word_indices, :].mean(axis=1)

            if self.args.norm_mention:
                mention_embs = normalize(mention_embs)

        if self.args.include_gram:
            mention_gram_embs = gram_embs[gram_indices, :].mean(axis=1)

            if self.args.norm_gram:
                mention_gram_embs = normalize(mention_gram_embs)

        if self.args.include_word:
            mention_word_embs = word_embs[word_indices, :].mean(axis=1)
            mention_word_embs = mention_word_embs @ W + b

            if self.args.norm_word:
                mention_word_embs = normalize(mention_word_embs)

        if self.args.include_context:
            mention_context_embs = word_embs[context_indices, :].mean(axis=1)
            mention_context_embs = mention_context_embs @ W + b

            if self.args.norm_word:
                mention_context_embs = normalize(mention_context_embs)

        if self.args.include_mention:
            mention_combined_embs = np.concatenate((mention_context_embs, mention_gram_embs, mention_embs), axis=1)
        else:
            # 100
            if self.args.include_gram and not self.args.include_context and not self.args.include_word:
                mention_combined_embs = mention_gram_embs
            # 001
            elif not self.args.include_gram and not self.args.include_context and self.args.include_word:
                mention_combined_embs = mention_word_embs
            # 010
            elif not self.args.include_gram and self.args.include_context and not self.args.include_word:
                mention_combined_embs = mention_context_embs
            # 110
            elif self.args.include_gram and self.args.include_context and not self.args.include_word:
                mention_combined_embs = np.concatenate((mention_context_embs, mention_gram_embs), axis=1)
            # 101
            elif self.args.include_gram and not self.args.include_context and self.args.include_word:
                mention_combined_embs = np.concatenate((mention_word_embs, mention_gram_embs), axis=1)
            # 011
            elif not self.args.include_gram and self.args.include_context and self.args.include_word:
                mention_combined_embs = np.concatenate((mention_context_embs, mention_word_embs), axis=1)
            # 111
            else:
                mention_combined_embs = np.concatenate((mention_context_embs, mention_gram_embs, mention_word_embs), axis=1)

        if self.args.norm_final:
            mention_combined_embs = normalize(mention_combined_embs)

        return mention_combined_embs

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

        params = self._get_model_params(model)
        ent_combined_embs = self._get_ent_combined_embs(params=params)
        wiki_mention_combined_embs = self._get_mention_combined_embs(params=params, data='wiki')
        conll_mention_combined_embs = self._get_mention_combined_embs(params=params, data='conll')

        if self.args.debug:
            print('Ent Shape : {}'.format(ent_combined_embs.shape))
            print('Wiki Mention Shape : {}'.format(wiki_mention_combined_embs.shape))
            print('Conll Mention Shape : {}'.format(conll_mention_combined_embs.shape))
            print('Wiki Gold Shape : {}'.format(self.wiki_all_gold[self.wiki_mask].shape))
            print('Conll Gold Shape : {}'.format(self.conll_all_gold.shape))
            print(ent_combined_embs[:5, :])
            print(wiki_mention_combined_embs[:5, :])
            print(conll_mention_combined_embs[:5, :])

        # Create / search in Faiss Index
        if self.args.measure == 'ip':
            index = faiss.IndexFlatIP(ent_combined_embs.shape[1])
        else:
            index = faiss.IndexFlatL2(ent_combined_embs.shape[1])
        index.add(ent_combined_embs)

        D_wiki, I_wiki = index.search(wiki_mention_combined_embs.astype(np.float32), 100)
        D_conll, I_conll = index.search(conll_mention_combined_embs.astype(np.float32), 100)

        if self.args.debug:
            print('Wiki result : {}'.format(I_wiki[:20, :10]))
            print('Conll result : {}'.format(I_conll[:20, :10]))

            wiki_debug_result = self._get_debug_string(I=I_wiki, data='wiki', result=True)
            conll_debug_result = self._get_debug_string(I=I_conll, data='conll', result=True)

            print("Wikipedia Debug Results")
            print(wiki_debug_result)

            print("Conll Debug Results")
            print(conll_debug_result)

        # Evaluate rankings
        top1_wiki, top10_wiki, top100_wiki, mrr_wiki = eval_ranking(I_wiki, self.wiki_all_gold[self.wiki_mask], [1, 10, 100])
        top1_conll, top10_conll, top100_conll, mrr_conll = eval_ranking(I_conll, self.conll_all_gold, [1, 10, 100])

        # Error analysis
        if error:
            print('Wiki Errors')
            check_errors(I_wiki, self.wiki_all_gold[self.wiki_mask], self.wiki_mention_word_indices[self.wiki_mask, :],
                         self.rev_ent_dict, self.rev_word_dict, [1, 10, 100])
            print('\n')
            print('Conll Errors')
            check_errors(I_conll, self.conll_all_gold, self.conll_mention_word_indices,
                         self.rev_ent_dict, self.rev_word_dict, [1, 10, 100])

        return top1_wiki, top10_wiki, top100_wiki, mrr_wiki, top1_conll, top10_conll, top100_conll, mrr_conll
